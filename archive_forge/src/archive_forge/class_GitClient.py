import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
class GitClient:
    """Git smart server client."""

    def __init__(self, thin_packs=True, report_activity=None, quiet=False, include_tags=False) -> None:
        """Create a new GitClient instance.

        Args:
          thin_packs: Whether or not thin packs should be retrieved
          report_activity: Optional callback for reporting transport
            activity.
          include_tags: send annotated tags when sending the objects they point
            to
        """
        self._report_activity = report_activity
        self._report_status_parser: Optional[ReportStatusParser] = None
        self._fetch_capabilities = set(UPLOAD_CAPABILITIES)
        self._fetch_capabilities.add(capability_agent())
        self._send_capabilities = set(RECEIVE_CAPABILITIES)
        self._send_capabilities.add(capability_agent())
        if quiet:
            self._send_capabilities.add(CAPABILITY_QUIET)
        if not thin_packs:
            self._fetch_capabilities.remove(CAPABILITY_THIN_PACK)
        if include_tags:
            self._fetch_capabilities.add(CAPABILITY_INCLUDE_TAG)

    def get_url(self, path):
        """Retrieves full url to given path.

        Args:
          path: Repository path (as string)

        Returns:
          Url to path (as string)

        """
        raise NotImplementedError(self.get_url)

    @classmethod
    def from_parsedurl(cls, parsedurl, **kwargs):
        """Create an instance of this client from a urlparse.parsed object.

        Args:
          parsedurl: Result of urlparse()

        Returns:
          A `GitClient` object
        """
        raise NotImplementedError(cls.from_parsedurl)

    def send_pack(self, path, update_refs, generate_pack_data: Callable[[Set[bytes], Set[bytes], bool], Tuple[int, Iterator[UnpackedObject]]], progress=None):
        """Upload a pack to a remote repository.

        Args:
          path: Repository path (as bytestring)
          update_refs: Function to determine changes to remote refs. Receive
            dict with existing remote refs, returns dict with
            changed refs (name -> sha, where sha=ZERO_SHA for deletions)
          generate_pack_data: Function that can return a tuple
            with number of objects and list of pack data to include
          progress: Optional progress function

        Returns:
          SendPackResult object

        Raises:
          SendPackError: if server rejects the pack data

        """
        raise NotImplementedError(self.send_pack)

    def clone(self, path, target_path, mkdir: bool=True, bare: bool=False, origin: Optional[str]='origin', checkout=None, branch=None, progress=None, depth=None) -> Repo:
        """Clone a repository."""
        from .refs import _set_default_branch, _set_head, _set_origin_head
        if mkdir:
            os.mkdir(target_path)
        try:
            target = None
            if not bare:
                target = Repo.init(target_path)
                if checkout is None:
                    checkout = True
            else:
                if checkout:
                    raise ValueError('checkout and bare are incompatible')
                target = Repo.init_bare(target_path)
            if isinstance(self, (LocalGitClient, SubprocessGitClient)):
                encoded_path = path.encode('utf-8')
            else:
                encoded_path = self.get_url(path).encode('utf-8')
            assert target is not None
            if origin is not None:
                target_config = target.get_config()
                target_config.set((b'remote', origin.encode('utf-8')), b'url', encoded_path)
                target_config.set((b'remote', origin.encode('utf-8')), b'fetch', b'+refs/heads/*:refs/remotes/' + origin.encode('utf-8') + b'/*')
                target_config.write_to_path()
            ref_message = b'clone: from ' + encoded_path
            result = self.fetch(path, target, progress=progress, depth=depth)
            if origin is not None:
                _import_remote_refs(target.refs, origin, result.refs, message=ref_message)
            origin_head = result.symrefs.get(b'HEAD')
            origin_sha = result.refs.get(b'HEAD')
            if origin is None or (origin_sha and (not origin_head)):
                target.refs[b'HEAD'] = origin_sha
                head = origin_sha
            else:
                _set_origin_head(target.refs, origin.encode('utf-8'), origin_head)
                head_ref = _set_default_branch(target.refs, origin.encode('utf-8'), origin_head, branch, ref_message)
                if head_ref:
                    head = _set_head(target.refs, head_ref, ref_message)
                else:
                    head = None
            if checkout and head is not None:
                target.reset_index()
        except BaseException:
            if target is not None:
                target.close()
            if mkdir:
                import shutil
                shutil.rmtree(target_path)
            raise
        return target

    def fetch(self, path: str, target: Repo, determine_wants: Optional[Callable[[Dict[bytes, bytes], Optional[int]], List[bytes]]]=None, progress: Optional[Callable[[bytes], None]]=None, depth: Optional[int]=None) -> FetchPackResult:
        """Fetch into a target repository.

        Args:
          path: Path to fetch from (as bytestring)
          target: Target repository to fetch into
          determine_wants: Optional function to determine what refs to fetch.
            Receives dictionary of name->sha, should return
            list of shas to fetch. Defaults to all shas.
          progress: Optional progress function
          depth: Depth to fetch at

        Returns:
          Dictionary with all remote refs (not just those fetched)

        """
        if determine_wants is None:
            determine_wants = target.object_store.determine_wants_all
        if CAPABILITY_THIN_PACK in self._fetch_capabilities:
            from tempfile import SpooledTemporaryFile
            f: IO[bytes] = SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE, prefix='incoming-', dir=getattr(target.object_store, 'path', None))

            def commit():
                if f.tell():
                    f.seek(0)
                    target.object_store.add_thin_pack(f.read, None, progress=progress)
                f.close()

            def abort():
                f.close()
        else:
            f, commit, abort = target.object_store.add_pack()
        try:
            result = self.fetch_pack(path, determine_wants, target.get_graph_walker(), f.write, progress=progress, depth=depth)
        except BaseException:
            abort()
            raise
        else:
            commit()
        target.update_shallow(result.new_shallow, result.new_unshallow)
        return result

    def fetch_pack(self, path: str, determine_wants, graph_walker, pack_data, *, progress: Optional[Callable[[bytes], None]]=None, depth: Optional[int]=None):
        """Retrieve a pack from a git smart server.

        Args:
          path: Remote path to fetch from
          determine_wants: Function determine what refs
            to fetch. Receives dictionary of name->sha, should return
            list of shas to fetch.
          graph_walker: Object with next() and ack().
          pack_data: Callback called for each bit of data in the pack
          progress: Callback for progress reports (strings)
          depth: Shallow fetch depth

        Returns:
          FetchPackResult object

        """
        raise NotImplementedError(self.fetch_pack)

    def get_refs(self, path):
        """Retrieve the current refs from a git smart server.

        Args:
          path: Path to the repo to fetch from. (as bytestring)
        """
        raise NotImplementedError(self.get_refs)

    @staticmethod
    def _should_send_pack(new_refs):
        return any((sha != ZERO_SHA for sha in new_refs.values()))

    def _negotiate_receive_pack_capabilities(self, server_capabilities):
        negotiated_capabilities = self._send_capabilities & server_capabilities
        agent = None
        for capability in server_capabilities:
            k, v = parse_capability(capability)
            if k == CAPABILITY_AGENT:
                agent = v
        extract_capability_names(server_capabilities) - KNOWN_RECEIVE_CAPABILITIES
        return (negotiated_capabilities, agent)

    def _handle_receive_pack_tail(self, proto: Protocol, capabilities: Set[bytes], progress: Optional[Callable[[bytes], None]]=None) -> Optional[Dict[bytes, Optional[str]]]:
        """Handle the tail of a 'git-receive-pack' request.

        Args:
          proto: Protocol object to read from
          capabilities: List of negotiated capabilities
          progress: Optional progress reporting function

        Returns:
          dict mapping ref name to:
            error message if the ref failed to update
            None if it was updated successfully
        """
        if CAPABILITY_SIDE_BAND_64K in capabilities:
            if progress is None:

                def progress(x):
                    pass
            if CAPABILITY_REPORT_STATUS in capabilities:
                assert self._report_status_parser is not None
                pktline_parser = PktLineParser(self._report_status_parser.handle_packet)
            for chan, data in _read_side_band64k_data(proto.read_pkt_seq()):
                if chan == SIDE_BAND_CHANNEL_DATA:
                    if CAPABILITY_REPORT_STATUS in capabilities:
                        pktline_parser.parse(data)
                elif chan == SIDE_BAND_CHANNEL_PROGRESS:
                    progress(data)
                else:
                    raise AssertionError('Invalid sideband channel %d' % chan)
        elif CAPABILITY_REPORT_STATUS in capabilities:
            assert self._report_status_parser
            for pkt in proto.read_pkt_seq():
                self._report_status_parser.handle_packet(pkt)
        if self._report_status_parser is not None:
            return dict(self._report_status_parser.check())
        return None

    def _negotiate_upload_pack_capabilities(self, server_capabilities):
        extract_capability_names(server_capabilities) - KNOWN_UPLOAD_CAPABILITIES
        symrefs = {}
        agent = None
        for capability in server_capabilities:
            k, v = parse_capability(capability)
            if k == CAPABILITY_SYMREF:
                src, dst = v.split(b':', 1)
                symrefs[src] = dst
            if k == CAPABILITY_AGENT:
                agent = v
        negotiated_capabilities = self._fetch_capabilities & server_capabilities
        return (negotiated_capabilities, symrefs, agent)

    def archive(self, path, committish, write_data, progress=None, write_error=None, format=None, subdirs=None, prefix=None):
        """Retrieve an archive of the specified tree."""
        raise NotImplementedError(self.archive)