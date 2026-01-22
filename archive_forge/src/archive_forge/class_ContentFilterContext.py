from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
class ContentFilterContext:
    """Object providing information that filters can use."""

    def __init__(self, relpath=None, tree=None):
        """Create a context.

        Args:
          relpath: the relative path or None if this context doesn't
           support that information.
          tree: the Tree providing this file or None if this context
           doesn't support that information.
        """
        self._relpath = relpath
        self._tree = tree
        self._revision_id = None
        self._revision = None

    def relpath(self):
        """Relative path of file to tree-root."""
        return self._relpath

    def source_tree(self):
        """Source Tree object."""
        return self._tree

    def revision_id(self):
        """Id of revision that last changed this file."""
        if self._revision_id is None:
            if self._tree is not None:
                self._revision_id = self._tree.get_file_revision(self._relpath)
        return self._revision_id

    def revision(self):
        """Revision this variation of the file was introduced in."""
        if self._revision is None:
            rev_id = self.revision_id()
            if rev_id is not None:
                repo = getattr(self._tree, '_repository', None)
                if repo is None:
                    repo = self._tree.branch.repository
                self._revision = repo.get_revision(rev_id)
        return self._revision