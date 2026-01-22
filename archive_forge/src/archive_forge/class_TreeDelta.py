from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
class TreeDelta:
    """Describes changes from one tree to another.

    Contains seven lists with TreeChange objects.

    added
    removed
    renamed
    copied
    kind_changed
    modified
    unchanged
    unversioned

    Each id is listed only once.

    Files that are both modified and renamed or copied are listed only in
    renamed or copied, with the text_modified flag true. The text_modified
    applies either to the content of the file or the target of the
    symbolic link, depending of the kind of file.

    Files are only considered renamed if their name has changed or
    their parent directory has changed.  Renaming a directory
    does not count as renaming all its contents.

    The lists are normally sorted when the delta is created.
    """

    def __init__(self):
        self.added = []
        self.removed = []
        self.renamed = []
        self.copied = []
        self.kind_changed = []
        self.modified = []
        self.unchanged = []
        self.unversioned = []
        self.missing = []

    def __eq__(self, other):
        if not isinstance(other, TreeDelta):
            return False
        return self.added == other.added and self.removed == other.removed and (self.renamed == other.renamed) and (self.copied == other.copied) and (self.modified == other.modified) and (self.unchanged == other.unchanged) and (self.kind_changed == other.kind_changed) and (self.unversioned == other.unversioned)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'TreeDelta(added=%r, removed=%r, renamed=%r, copied=%r, kind_changed=%r, modified=%r, unchanged=%r, unversioned=%r)' % (self.added, self.removed, self.renamed, self.copied, self.kind_changed, self.modified, self.unchanged, self.unversioned)

    def has_changed(self):
        return bool(self.modified or self.added or self.removed or self.renamed or self.copied or self.kind_changed)

    def get_changes_as_text(self, show_ids=False, show_unchanged=False, short_status=False):
        output = StringIO()
        report_delta(output, self, short_status, show_ids, show_unchanged)
        return output.getvalue()