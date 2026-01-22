from typing import List, Optional, Union
from . import errors, hooks, osutils, trace, tree
class MutableTree(tree.Tree):
    """A MutableTree is a specialisation of Tree which is able to be mutated.

    Generally speaking these mutations are only possible within a lock_write
    context, and will revert if the lock is broken abnormally - but this cannot
    be guaranteed - depending on the exact implementation of the mutable state.

    The most common form of Mutable Tree is WorkingTree, see breezy.workingtree.
    For tests we also have MemoryTree which is a MutableTree whose contents are
    entirely in memory.

    For now, we are not treating MutableTree as an interface to provide
    conformance tests for - rather we are testing MemoryTree specifically, and
    interface testing implementations of WorkingTree.

    A mutable tree always has an associated Branch and ControlDir object - the
    branch and bzrdir attributes.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.case_sensitive = True

    def is_control_filename(self, filename):
        """True if filename is the name of a control file in this tree.

        :param filename: A filename within the tree. This is a relative path
            from the root of this tree.

        This is true IF and ONLY IF the filename is part of the meta data
        that bzr controls in this tree. I.E. a random .bzr directory placed
        on disk will not be a control file for this tree.
        """
        raise NotImplementedError(self.is_control_filename)

    def add(self, files: Union[str, List[str]], kinds: Optional[Union[str, List[str]]]=None):
        """Add paths to the set of versioned paths.

        Note that the command line normally calls smart_add instead,
        which can automatically recurse.

        This adds the files to the tree, so that they will be
        recorded by the next commit.

        Args:
          files: List of paths to add, relative to the base of the tree.
          kinds: Optional parameter to specify the kinds to be used for
            each file.
        """
        raise NotImplementedError(self.add)

    def add_reference(self, sub_tree):
        """Add a TreeReference to the tree, pointing at sub_tree.

        :param sub_tree: subtree to add.
        """
        raise errors.UnsupportedOperation(self.add_reference, self)

    def commit(self, message=None, revprops=None, *args, **kwargs):
        from breezy import commit
        possible_master_transports = []
        with self.lock_write():
            revprops = commit.Commit.update_revprops(revprops, self.branch, kwargs.pop('authors', None), kwargs.get('local', False), possible_master_transports)
            args = (message,) + args
            for hook in MutableTree.hooks['start_commit']:
                hook(self)
            committed_id = commit.Commit().commit(*args, working_tree=self, revprops=revprops, possible_master_transports=possible_master_transports, **kwargs)
            post_hook_params = PostCommitHookParams(self)
            for hook in MutableTree.hooks['post_commit']:
                hook(post_hook_params)
            return committed_id

    def has_changes(self, _from_tree=None):
        """Quickly check that the tree contains at least one commitable change.

        :param _from_tree: tree to compare against to find changes (default to
            the basis tree and is intended to be used by tests).

        :return: True if a change is found. False otherwise
        """
        raise NotImplementedError(self.has_changes)

    def check_changed_or_out_of_date(self, strict, opt_name, more_error, more_warning):
        """Check the tree for uncommitted changes and branch synchronization.

        If strict is None and not set in the config files, a warning is issued.
        If strict is True, an error is raised.
        If strict is False, no checks are done and no warning is issued.

        :param strict: True, False or None, searched in branch config if None.

        :param opt_name: strict option name to search in config file.

        :param more_error: Details about how to avoid the check.

        :param more_warning: Details about what is happening.
        """
        with self.lock_read():
            if strict is None:
                strict = self.branch.get_config_stack().get(opt_name)
            if strict is not False:
                err_class = None
                if self.has_changes():
                    err_class = errors.UncommittedChanges
                elif self.last_revision() != self.branch.last_revision():
                    err_class = errors.OutOfDateTree
                if err_class is not None:
                    if strict is None:
                        err = err_class(self, more=more_warning)
                        trace.warning('%s', err._format())
                    else:
                        err = err_class(self, more=more_error)
                        raise err

    def last_revision(self):
        """Return the revision id of the last commit performed in this tree.

        In early tree formats the result of last_revision is the same as the
        branch last_revision, but that is no longer the case for modern tree
        formats.

        last_revision returns the left most parent id, or None if there are no
        parents.

        last_revision was deprecated as of 0.11. Please use get_parent_ids
        instead.
        """
        raise NotImplementedError(self.last_revision)

    def lock_tree_write(self):
        """Lock the working tree for write, and the branch for read.

        This is useful for operations which only need to mutate the working
        tree. Taking out branch write locks is a relatively expensive process
        and may fail if the branch is on read only media. So branch write locks
        should only be taken out when we are modifying branch data - such as in
        operations like commit, pull, uncommit and update.
        """
        raise NotImplementedError(self.lock_tree_write)

    def lock_write(self):
        """Lock the tree and its branch. This allows mutating calls to be made.

        Some mutating methods will take out implicit write locks, but in
        general you should always obtain a write lock before calling mutating
        methods on a tree.
        """
        raise NotImplementedError(self.lock_write)

    def mkdir(self, path):
        """Create a directory in the tree.

        :param path: A unicode file path.
        :return: the file id of the new directory.
        """
        raise NotImplementedError(self.mkdir)

    def _observed_sha1(self, path, sha_and_stat):
        """Tell the tree we have observed a paths sha1.

        The intent of this function is to allow trees that have a hashcache to
        update the hashcache during commit. If the observed file is too new
        (based on the stat_value) to be safely hash-cached the tree will ignore
        it.

        The default implementation does nothing.

        :param path: The file path
        :param sha_and_stat: The sha 1 and stat result observed.
        :return: None
        """

    def put_file_bytes_non_atomic(self, path, bytes):
        """Update the content of a file in the tree.

        Note that the file is written in-place rather than being
        written to a temporary location and renamed. As a consequence,
        readers can potentially see the file half-written.

        :param path: path of the file
        :param bytes: the new file contents
        """
        raise NotImplementedError(self.put_file_bytes_non_atomic)

    def set_parent_ids(self, revision_ids, allow_leftmost_as_ghost=False):
        """Set the parents ids of the working tree.

        :param revision_ids: A list of revision_ids.
        """
        raise NotImplementedError(self.set_parent_ids)

    def set_parent_trees(self, parents_list, allow_leftmost_as_ghost=False):
        """Set the parents of the working tree.

        :param parents_list: A list of (revision_id, tree) tuples.
            If tree is None, then that element is treated as an unreachable
            parent tree - i.e. a ghost.
        """
        raise NotImplementedError(self.set_parent_trees)

    def smart_add(self, file_list, recurse=True, action=None, save=True):
        """Version file_list, optionally recursing into directories.

        This is designed more towards DWIM for humans than API clarity.
        For the specific behaviour see the help for cmd_add().

        :param file_list: List of zero or more paths.  *NB: these are
            interpreted relative to the process cwd, not relative to the
            tree.*  (Add and most other tree methods use tree-relative
            paths.)
        :param action: A reporter to be called with the working tree, parent_ie,
            path and kind of the path being added. It may return a file_id if
            a specific one should be used.
        :param save: Save the changes after completing the adds. If False
            this provides dry-run functionality by doing the add and not saving
            the changes.
        :return: A tuple - files_added, ignored_files. files_added is the count
            of added files, and ignored_files is a dict mapping files that were
            ignored to the rule that caused them to be ignored.
        """
        raise NotImplementedError(self.smart_add)

    def rename_one(self, from_rel, to_rel, after=False):
        """Rename one file.

        This can change the directory or the filename or both.

        rename_one has several 'modes' to work. First, it can rename a physical
        file and change the file_id. That is the normal mode. Second, it can
        only change the file_id without touching any physical file.

        rename_one uses the second mode if 'after == True' and 'to_rel' is
        either not versioned or newly added, and present in the working tree.

        rename_one uses the second mode if 'after == False' and 'from_rel' is
        versioned but no longer in the working tree, and 'to_rel' is not
        versioned but present in the working tree.

        rename_one uses the first mode if 'after == False' and 'from_rel' is
        versioned and present in the working tree, and 'to_rel' is not
        versioned and not present in the working tree.

        Everything else results in an error.
        """
        raise NotImplementedError(self.rename_one)

    def copy_one(self, from_rel, to_rel):
        """Copy one file or directory.

        This can change the directory or the filename or both.

        """
        raise NotImplementedError(self.copy_one)

    def transform(self, pb=None):
        """Return a transform object for use with this tree."""
        raise NotImplementedError(self.transform)