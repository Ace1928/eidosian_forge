import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
class BaseRepo:
    """Base class for a git repository.

    This base class is meant to be used for Repository implementations that e.g.
    work on top of a different transport than a standard filesystem path.

    Attributes:
      object_store: Dictionary-like object for accessing
        the objects
      refs: Dictionary-like object with the refs in this
        repository
    """

    def __init__(self, object_store: PackBasedObjectStore, refs: RefsContainer) -> None:
        """Open a repository.

        This shouldn't be called directly, but rather through one of the
        base classes, such as MemoryRepo or Repo.

        Args:
          object_store: Object store to use
          refs: Refs container to use
        """
        self.object_store = object_store
        self.refs = refs
        self._graftpoints: Dict[bytes, List[bytes]] = {}
        self.hooks: Dict[str, Hook] = {}

    def _determine_file_mode(self) -> bool:
        """Probe the file-system to determine whether permissions can be trusted.

        Returns: True if permissions can be trusted, False otherwise.
        """
        raise NotImplementedError(self._determine_file_mode)

    def _determine_symlinks(self) -> bool:
        """Probe the filesystem to determine whether symlinks can be created.

        Returns: True if symlinks can be created, False otherwise.
        """
        return sys.platform != 'win32'

    def _init_files(self, bare: bool, symlinks: Optional[bool]=None) -> None:
        """Initialize a default set of named files."""
        from .config import ConfigFile
        self._put_named_file('description', b'Unnamed repository')
        f = BytesIO()
        cf = ConfigFile()
        cf.set('core', 'repositoryformatversion', '0')
        if self._determine_file_mode():
            cf.set('core', 'filemode', True)
        else:
            cf.set('core', 'filemode', False)
        if symlinks is None and (not bare):
            symlinks = self._determine_symlinks()
        if symlinks is False:
            cf.set('core', 'symlinks', symlinks)
        cf.set('core', 'bare', bare)
        cf.set('core', 'logallrefupdates', True)
        cf.write_to_file(f)
        self._put_named_file('config', f.getvalue())
        self._put_named_file(os.path.join('info', 'exclude'), b'')

    def get_named_file(self, path: str) -> Optional[BinaryIO]:
        """Get a file from the control dir with a specific name.

        Although the filename should be interpreted as a filename relative to
        the control dir in a disk-based Repo, the object returned need not be
        pointing to a file in that location.

        Args:
          path: The path to the file, relative to the control dir.
        Returns: An open file object, or None if the file does not exist.
        """
        raise NotImplementedError(self.get_named_file)

    def _put_named_file(self, path: str, contents: bytes):
        """Write a file to the control dir with the given name and contents.

        Args:
          path: The path to the file, relative to the control dir.
          contents: A string to write to the file.
        """
        raise NotImplementedError(self._put_named_file)

    def _del_named_file(self, path: str):
        """Delete a file in the control directory with the given name."""
        raise NotImplementedError(self._del_named_file)

    def open_index(self) -> 'Index':
        """Open the index for this repository.

        Raises:
          NoIndexPresent: If no index is present
        Returns: The matching `Index`
        """
        raise NotImplementedError(self.open_index)

    def fetch(self, target, determine_wants=None, progress=None, depth=None):
        """Fetch objects into another repository.

        Args:
          target: The target repository
          determine_wants: Optional function to determine what refs to
            fetch.
          progress: Optional progress function
          depth: Optional shallow fetch depth
        Returns: The local refs
        """
        if determine_wants is None:
            determine_wants = target.object_store.determine_wants_all
        count, pack_data = self.fetch_pack_data(determine_wants, target.get_graph_walker(), progress=progress, depth=depth)
        target.object_store.add_pack_data(count, pack_data, progress)
        return self.get_refs()

    def fetch_pack_data(self, determine_wants, graph_walker, progress, get_tagged=None, depth=None):
        """Fetch the pack data required for a set of revisions.

        Args:
          determine_wants: Function that takes a dictionary with heads
            and returns the list of heads to fetch.
          graph_walker: Object that can iterate over the list of revisions
            to fetch and has an "ack" method that will be called to acknowledge
            that a revision is present.
          progress: Simple progress function that will be called with
            updated progress strings.
          get_tagged: Function that returns a dict of pointed-to sha ->
            tag sha for including tags.
          depth: Shallow fetch depth
        Returns: count and iterator over pack data
        """
        missing_objects = self.find_missing_objects(determine_wants, graph_walker, progress, get_tagged, depth=depth)
        remote_has = missing_objects.get_remote_has()
        object_ids = list(missing_objects)
        return (len(object_ids), generate_unpacked_objects(self.object_store, object_ids, progress=progress, other_haves=remote_has))

    def find_missing_objects(self, determine_wants, graph_walker, progress, get_tagged=None, depth=None) -> Optional[MissingObjectFinder]:
        """Fetch the missing objects required for a set of revisions.

        Args:
          determine_wants: Function that takes a dictionary with heads
            and returns the list of heads to fetch.
          graph_walker: Object that can iterate over the list of revisions
            to fetch and has an "ack" method that will be called to acknowledge
            that a revision is present.
          progress: Simple progress function that will be called with
            updated progress strings.
          get_tagged: Function that returns a dict of pointed-to sha ->
            tag sha for including tags.
          depth: Shallow fetch depth
        Returns: iterator over objects, with __len__ implemented
        """
        if depth not in (None, 0):
            raise NotImplementedError('depth not supported yet')
        refs = serialize_refs(self.object_store, self.get_refs())
        wants = determine_wants(refs)
        if not isinstance(wants, list):
            raise TypeError('determine_wants() did not return a list')
        shallows: FrozenSet[ObjectID] = getattr(graph_walker, 'shallow', frozenset())
        unshallows: FrozenSet[ObjectID] = getattr(graph_walker, 'unshallow', frozenset())
        if wants == []:
            if shallows or unshallows:
                return None

            class DummyMissingObjectFinder:

                def get_remote_has(self):
                    return None

                def __len__(self) -> int:
                    return 0

                def __iter__(self):
                    yield from []
            return DummyMissingObjectFinder()
        haves = self.object_store.find_common_revisions(graph_walker)
        if shallows or unshallows:
            haves = []
        parents_provider = ParentsProvider(self.object_store, shallows=shallows)

        def get_parents(commit):
            return parents_provider.get_parents(commit.id, commit)
        return MissingObjectFinder(self.object_store, haves=haves, wants=wants, shallow=self.get_shallow(), progress=progress, get_tagged=get_tagged, get_parents=get_parents)

    def generate_pack_data(self, have: List[ObjectID], want: List[ObjectID], progress: Optional[Callable[[str], None]]=None, ofs_delta: Optional[bool]=None):
        """Generate pack data objects for a set of wants/haves.

        Args:
          have: List of SHA1s of objects that should not be sent
          want: List of SHA1s of objects that should be sent
          ofs_delta: Whether OFS deltas can be included
          progress: Optional progress reporting method
        """
        return self.object_store.generate_pack_data(have, want, shallow=self.get_shallow(), progress=progress, ofs_delta=ofs_delta)

    def get_graph_walker(self, heads: Optional[List[ObjectID]]=None) -> ObjectStoreGraphWalker:
        """Retrieve a graph walker.

        A graph walker is used by a remote repository (or proxy)
        to find out which objects are present in this repository.

        Args:
          heads: Repository heads to use (optional)
        Returns: A graph walker object
        """
        if heads is None:
            heads = [sha for sha in self.refs.as_dict(b'refs/heads').values() if sha in self.object_store]
        parents_provider = ParentsProvider(self.object_store)
        return ObjectStoreGraphWalker(heads, parents_provider.get_parents, shallow=self.get_shallow())

    def get_refs(self) -> Dict[bytes, bytes]:
        """Get dictionary with all refs.

        Returns: A ``dict`` mapping ref names to SHA1s
        """
        return self.refs.as_dict()

    def head(self) -> bytes:
        """Return the SHA1 pointed at by HEAD."""
        return self.refs[b'HEAD']

    def _get_object(self, sha, cls):
        assert len(sha) in (20, 40)
        ret = self.get_object(sha)
        if not isinstance(ret, cls):
            if cls is Commit:
                raise NotCommitError(ret)
            elif cls is Blob:
                raise NotBlobError(ret)
            elif cls is Tree:
                raise NotTreeError(ret)
            elif cls is Tag:
                raise NotTagError(ret)
            else:
                raise Exception(f'Type invalid: {ret.type_name!r} != {cls.type_name!r}')
        return ret

    def get_object(self, sha: bytes) -> ShaFile:
        """Retrieve the object with the specified SHA.

        Args:
          sha: SHA to retrieve
        Returns: A ShaFile object
        Raises:
          KeyError: when the object can not be found
        """
        return self.object_store[sha]

    def parents_provider(self) -> ParentsProvider:
        return ParentsProvider(self.object_store, grafts=self._graftpoints, shallows=self.get_shallow())

    def get_parents(self, sha: bytes, commit: Optional[Commit]=None) -> List[bytes]:
        """Retrieve the parents of a specific commit.

        If the specific commit is a graftpoint, the graft parents
        will be returned instead.

        Args:
          sha: SHA of the commit for which to retrieve the parents
          commit: Optional commit matching the sha
        Returns: List of parents
        """
        return self.parents_provider().get_parents(sha, commit)

    def get_config(self) -> 'ConfigFile':
        """Retrieve the config object.

        Returns: `ConfigFile` object for the ``.git/config`` file.
        """
        raise NotImplementedError(self.get_config)

    def get_worktree_config(self) -> 'ConfigFile':
        """Retrieve the worktree config object."""
        raise NotImplementedError(self.get_worktree_config)

    def get_description(self):
        """Retrieve the description for this repository.

        Returns: String with the description of the repository
            as set by the user.
        """
        raise NotImplementedError(self.get_description)

    def set_description(self, description):
        """Set the description for this repository.

        Args:
          description: Text to set as description for this repository.
        """
        raise NotImplementedError(self.set_description)

    def get_config_stack(self) -> 'StackedConfig':
        """Return a config stack for this repository.

        This stack accesses the configuration for both this repository
        itself (.git/config) and the global configuration, which usually
        lives in ~/.gitconfig.

        Returns: `Config` instance for this repository
        """
        from .config import ConfigFile, StackedConfig
        local_config = self.get_config()
        backends: List[ConfigFile] = [local_config]
        if local_config.get_boolean((b'extensions',), b'worktreeconfig', False):
            backends.append(self.get_worktree_config())
        backends += StackedConfig.default_backends()
        return StackedConfig(backends, writable=local_config)

    def get_shallow(self) -> Set[ObjectID]:
        """Get the set of shallow commits.

        Returns: Set of shallow commits.
        """
        f = self.get_named_file('shallow')
        if f is None:
            return set()
        with f:
            return {line.strip() for line in f}

    def update_shallow(self, new_shallow, new_unshallow):
        """Update the list of shallow objects.

        Args:
          new_shallow: Newly shallow objects
          new_unshallow: Newly no longer shallow objects
        """
        shallow = self.get_shallow()
        if new_shallow:
            shallow.update(new_shallow)
        if new_unshallow:
            shallow.difference_update(new_unshallow)
        if shallow:
            self._put_named_file('shallow', b''.join([sha + b'\n' for sha in shallow]))
        else:
            self._del_named_file('shallow')

    def get_peeled(self, ref: Ref) -> ObjectID:
        """Get the peeled value of a ref.

        Args:
          ref: The refname to peel.
        Returns: The fully-peeled SHA1 of a tag object, after peeling all
            intermediate tags; if the original ref does not point to a tag,
            this will equal the original SHA1.
        """
        cached = self.refs.get_peeled(ref)
        if cached is not None:
            return cached
        return peel_sha(self.object_store, self.refs[ref])[1].id

    def get_walker(self, include: Optional[List[bytes]]=None, *args, **kwargs):
        """Obtain a walker for this repository.

        Args:
          include: Iterable of SHAs of commits to include along with their
            ancestors. Defaults to [HEAD]
          exclude: Iterable of SHAs of commits to exclude along with their
            ancestors, overriding includes.
          order: ORDER_* constant specifying the order of results.
            Anything other than ORDER_DATE may result in O(n) memory usage.
          reverse: If True, reverse the order of output, requiring O(n)
            memory.
          max_entries: The maximum number of entries to yield, or None for
            no limit.
          paths: Iterable of file or subtree paths to show entries for.
          rename_detector: diff.RenameDetector object for detecting
            renames.
          follow: If True, follow path across renames/copies. Forces a
            default rename_detector.
          since: Timestamp to list commits after.
          until: Timestamp to list commits before.
          queue_cls: A class to use for a queue of commits, supporting the
            iterator protocol. The constructor takes a single argument, the
            Walker.
        Returns: A `Walker` object
        """
        from .walk import Walker
        if include is None:
            include = [self.head()]
        kwargs['get_parents'] = lambda commit: self.get_parents(commit.id, commit)
        return Walker(self.object_store, include, *args, **kwargs)

    def __getitem__(self, name: Union[ObjectID, Ref]):
        """Retrieve a Git object by SHA1 or ref.

        Args:
          name: A Git object SHA1 or a ref name
        Returns: A `ShaFile` object, such as a Commit or Blob
        Raises:
          KeyError: when the specified ref or object does not exist
        """
        if not isinstance(name, bytes):
            raise TypeError("'name' must be bytestring, not %.80s" % type(name).__name__)
        if len(name) in (20, 40):
            try:
                return self.object_store[name]
            except (KeyError, ValueError):
                pass
        try:
            return self.object_store[self.refs[name]]
        except RefFormatError as exc:
            raise KeyError(name) from exc

    def __contains__(self, name: bytes) -> bool:
        """Check if a specific Git object or ref is present.

        Args:
          name: Git object SHA1 or ref name
        """
        if len(name) == 20 or (len(name) == 40 and valid_hexsha(name)):
            return name in self.object_store or name in self.refs
        else:
            return name in self.refs

    def __setitem__(self, name: bytes, value: Union[ShaFile, bytes]) -> None:
        """Set a ref.

        Args:
          name: ref name
          value: Ref value - either a ShaFile object, or a hex sha
        """
        if name.startswith(b'refs/') or name == b'HEAD':
            if isinstance(value, ShaFile):
                self.refs[name] = value.id
            elif isinstance(value, bytes):
                self.refs[name] = value
            else:
                raise TypeError(value)
        else:
            raise ValueError(name)

    def __delitem__(self, name: bytes) -> None:
        """Remove a ref.

        Args:
          name: Name of the ref to remove
        """
        if name.startswith(b'refs/') or name == b'HEAD':
            del self.refs[name]
        else:
            raise ValueError(name)

    def _get_user_identity(self, config: 'StackedConfig', kind: Optional[str]=None) -> bytes:
        """Determine the identity to use for new commits."""
        warnings.warn('use get_user_identity() rather than Repo._get_user_identity', DeprecationWarning)
        return get_user_identity(config)

    def _add_graftpoints(self, updated_graftpoints: Dict[bytes, List[bytes]]):
        """Add or modify graftpoints.

        Args:
          updated_graftpoints: Dict of commit shas to list of parent shas
        """
        for commit, parents in updated_graftpoints.items():
            for sha in [commit, *parents]:
                check_hexsha(sha, 'Invalid graftpoint')
        self._graftpoints.update(updated_graftpoints)

    def _remove_graftpoints(self, to_remove: List[bytes]=[]) -> None:
        """Remove graftpoints.

        Args:
          to_remove: List of commit shas
        """
        for sha in to_remove:
            del self._graftpoints[sha]

    def _read_heads(self, name):
        f = self.get_named_file(name)
        if f is None:
            return []
        with f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def do_commit(self, message: Optional[bytes]=None, committer: Optional[bytes]=None, author: Optional[bytes]=None, commit_timestamp=None, commit_timezone=None, author_timestamp=None, author_timezone=None, tree: Optional[ObjectID]=None, encoding: Optional[bytes]=None, ref: Ref=b'HEAD', merge_heads: Optional[List[ObjectID]]=None, no_verify: bool=False, sign: bool=False):
        """Create a new commit.

        If not specified, committer and author default to
        get_user_identity(..., 'COMMITTER')
        and get_user_identity(..., 'AUTHOR') respectively.

        Args:
          message: Commit message
          committer: Committer fullname
          author: Author fullname
          commit_timestamp: Commit timestamp (defaults to now)
          commit_timezone: Commit timestamp timezone (defaults to GMT)
          author_timestamp: Author timestamp (defaults to commit
            timestamp)
          author_timezone: Author timestamp timezone
            (defaults to commit timestamp timezone)
          tree: SHA1 of the tree root to use (if not specified the
            current index will be committed).
          encoding: Encoding
          ref: Optional ref to commit to (defaults to current branch)
          merge_heads: Merge heads (defaults to .git/MERGE_HEAD)
          no_verify: Skip pre-commit and commit-msg hooks
          sign: GPG Sign the commit (bool, defaults to False,
            pass True to use default GPG key,
            pass a str containing Key ID to use a specific GPG key)

        Returns:
          New commit SHA1
        """
        try:
            if not no_verify:
                self.hooks['pre-commit'].execute()
        except HookError as exc:
            raise CommitError(exc) from exc
        except KeyError:
            pass
        c = Commit()
        if tree is None:
            index = self.open_index()
            c.tree = index.commit(self.object_store)
        else:
            if len(tree) != 40:
                raise ValueError('tree must be a 40-byte hex sha string')
            c.tree = tree
        config = self.get_config_stack()
        if merge_heads is None:
            merge_heads = self._read_heads('MERGE_HEAD')
        if committer is None:
            committer = get_user_identity(config, kind='COMMITTER')
        check_user_identity(committer)
        c.committer = committer
        if commit_timestamp is None:
            commit_timestamp = time.time()
        c.commit_time = int(commit_timestamp)
        if commit_timezone is None:
            commit_timezone = 0
        c.commit_timezone = commit_timezone
        if author is None:
            author = get_user_identity(config, kind='AUTHOR')
        c.author = author
        check_user_identity(author)
        if author_timestamp is None:
            author_timestamp = commit_timestamp
        c.author_time = int(author_timestamp)
        if author_timezone is None:
            author_timezone = commit_timezone
        c.author_timezone = author_timezone
        if encoding is None:
            try:
                encoding = config.get(('i18n',), 'commitEncoding')
            except KeyError:
                pass
        if encoding is not None:
            c.encoding = encoding
        if message is None:
            raise ValueError('No commit message specified')
        try:
            if no_verify:
                c.message = message
            else:
                c.message = self.hooks['commit-msg'].execute(message)
                if c.message is None:
                    c.message = message
        except HookError as exc:
            raise CommitError(exc) from exc
        except KeyError:
            c.message = message
        keyid = sign if isinstance(sign, str) else None
        if ref is None:
            c.parents = merge_heads
            if sign:
                c.sign(keyid)
            self.object_store.add_object(c)
        else:
            try:
                old_head = self.refs[ref]
                c.parents = [old_head, *merge_heads]
                if sign:
                    c.sign(keyid)
                self.object_store.add_object(c)
                ok = self.refs.set_if_equals(ref, old_head, c.id, message=b'commit: ' + message, committer=committer, timestamp=commit_timestamp, timezone=commit_timezone)
            except KeyError:
                c.parents = merge_heads
                if sign:
                    c.sign(keyid)
                self.object_store.add_object(c)
                ok = self.refs.add_if_new(ref, c.id, message=b'commit: ' + message, committer=committer, timestamp=commit_timestamp, timezone=commit_timezone)
            if not ok:
                raise CommitError(f'{ref!r} changed during commit')
        self._del_named_file('MERGE_HEAD')
        try:
            self.hooks['post-commit'].execute()
        except HookError as e:
            warnings.warn('post-commit hook failed: %s' % e, UserWarning)
        except KeyError:
            pass
        return c.id