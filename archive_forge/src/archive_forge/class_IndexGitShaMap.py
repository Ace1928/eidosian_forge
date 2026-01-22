import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
class IndexGitShaMap(GitShaMap):
    """SHA Map that uses the Bazaar APIs to store a cache.

    BTree Index file with the following contents:

    ("git", <sha1>, "X") -> "<type> <type-data1> <type-data2>"
    ("commit", <revid>, "X") -> "<sha1> <tree-id>"
    ("blob", <fileid>, <revid>) -> <sha1>

    """

    def __init__(self, transport=None):
        self._name = None
        if transport is None:
            self._transport = None
            self._index = _mod_index.InMemoryGraphIndex(0, key_elements=3)
            self._builder = self._index
        else:
            self._builder = None
            self._transport = transport
            self._index = _mod_index.CombinedGraphIndex([])
            for name in self._transport.list_dir('.'):
                if not name.endswith('.rix'):
                    continue
                x = _mod_btree_index.BTreeGraphIndex(self._transport, name, self._transport.stat(name).st_size)
                self._index.insert_index(0, x)

    @classmethod
    def from_repository(cls, repository):
        transport = getattr(repository, '_transport', None)
        if transport is not None:
            try:
                transport.mkdir('git')
            except FileExists:
                pass
            return cls(transport.clone('git'))
        return cls(get_transport_from_path(get_cache_dir()))

    def __repr__(self):
        if self._transport is not None:
            return '{}({!r})'.format(self.__class__.__name__, self._transport.base)
        else:
            return '%s()' % self.__class__.__name__

    def repack(self):
        if self._builder is not None:
            raise bzr_errors.BzrError('builder already open')
        self.start_write_group()
        self._builder.add_nodes(((key, value) for _, key, value in self._index.iter_all_entries()))
        to_remove = []
        for name in self._transport.list_dir('.'):
            if name.endswith('.rix'):
                to_remove.append(name)
        self.commit_write_group()
        del self._index.indices[1:]
        for name in to_remove:
            self._transport.rename(name, name + '.old')

    def start_write_group(self):
        if self._builder is not None:
            raise bzr_errors.BzrError('builder already open')
        self._builder = _mod_btree_index.BTreeBuilder(0, key_elements=3)
        self._name = osutils.sha()

    def commit_write_group(self):
        if self._builder is None:
            raise bzr_errors.BzrError('builder not open')
        stream = self._builder.finish()
        name = self._name.hexdigest() + '.rix'
        size = self._transport.put_file(name, stream)
        index = _mod_btree_index.BTreeGraphIndex(self._transport, name, size)
        self._index.insert_index(0, index)
        self._builder = None
        self._name = None

    def abort_write_group(self):
        if self._builder is None:
            raise bzr_errors.BzrError('builder not open')
        self._builder = None
        self._name = None

    def _add_node(self, key, value):
        try:
            self._get_entry(key)
        except KeyError:
            self._builder.add_node(key, value)
            return False
        else:
            return True

    def _get_entry(self, key):
        entries = self._index.iter_entries([key])
        try:
            return next(entries)[2]
        except StopIteration:
            if self._builder is None:
                raise KeyError
            entries = self._builder.iter_entries([key])
            try:
                return next(entries)[2]
            except StopIteration:
                raise KeyError

    def _iter_entries_prefix(self, prefix):
        for entry in self._index.iter_entries_prefix([prefix]):
            yield (entry[1], entry[2])
        if self._builder is not None:
            for entry in self._builder.iter_entries_prefix([prefix]):
                yield (entry[1], entry[2])

    def lookup_commit(self, revid):
        return self._get_entry((b'commit', revid, b'X'))[:40]

    def _add_git_sha(self, hexsha, type, type_data):
        if hexsha is not None:
            self._name.update(hexsha)
            if type == b'commit':
                td = (type_data[0], type_data[1])
                try:
                    td += (type_data[2]['testament3-sha1'],)
                except KeyError:
                    pass
            else:
                td = type_data
            self._add_node((b'git', hexsha, b'X'), b' '.join((type,) + td))
        else:
            self._name.update(type + b' '.join(type_data))

    def lookup_blob_id(self, fileid, revision):
        return self._get_entry((b'blob', fileid, revision))

    def lookup_git_sha(self, sha):
        if len(sha) == 20:
            sha = sha_to_hex(sha)
        value = self._get_entry((b'git', sha, b'X'))
        data = value.split(b' ', 3)
        if data[0] == b'commit':
            try:
                if data[3]:
                    verifiers = {'testament3-sha1': data[3]}
                else:
                    verifiers = {}
            except IndexError:
                verifiers = {}
            yield ('commit', (data[1], data[2], verifiers))
        else:
            yield (data[0].decode('ascii'), tuple(data[1:]))

    def revids(self):
        """List the revision ids known."""
        for key, value in self._iter_entries_prefix((b'commit', None, None)):
            yield key[1]

    def missing_revisions(self, revids):
        """Return set of all the revisions that are not present."""
        missing_revids = set(revids)
        for _, key, value in self._index.iter_entries(((b'commit', revid, b'X') for revid in revids)):
            missing_revids.remove(key[1])
        return missing_revids

    def sha1s(self):
        """List the SHA1s."""
        for key, value in self._iter_entries_prefix((b'git', None, None)):
            yield key[1]