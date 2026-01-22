from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class HasLayout(Matcher):
    """A matcher that checks if a tree has a specific layout.

    :ivar entries: List of expected entries, as (path, file_id) pairs.
    """

    def __init__(self, entries):
        Matcher.__init__(self)
        self.entries = entries

    def get_tree_layout(self, tree, include_file_ids):
        """Get the (path, file_id) pairs for the current tree."""
        with tree.lock_read():
            for path, ie in tree.iter_entries_by_dir():
                if path != '':
                    path += ie.kind_character()
                if include_file_ids:
                    yield (path, ie.file_id)
                else:
                    yield path

    @staticmethod
    def _strip_unreferenced_directories(entries):
        """Strip all directories that don't (in)directly contain any files.

        :param entries: List of path strings or (path, ie) tuples to process
        """
        directories = []
        for entry in entries:
            if isinstance(entry, str):
                path = entry
            else:
                path = entry[0]
            if not path or path[-1] == '/':
                directories.append((path, entry))
            else:
                for dirpath, direntry in directories:
                    if osutils.is_inside(dirpath, path):
                        yield direntry
                directories = []
                yield entry

    def __str__(self):
        return 'HasLayout(%r)' % self.entries

    def match(self, tree):
        include_file_ids = self.entries and (not isinstance(self.entries[0], str))
        actual = list(self.get_tree_layout(tree, include_file_ids=include_file_ids))
        if not tree.has_versioned_directories():
            entries = list(self._strip_unreferenced_directories(self.entries))
        else:
            entries = self.entries
        return Equals(entries).match(actual)