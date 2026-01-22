import os.path
from ._paml import Paml
from . import _parse_codeml
def _set_rel_paths(self):
    """Make all file/directory paths relative to the PWD (PRIVATE).

        CODEML requires that all paths specified in the control file be
        relative to the directory from which it is called rather than
        absolute paths.
        """
    Paml._set_rel_paths(self)
    if self.tree is not None:
        self._rel_tree = os.path.relpath(self.tree, self.working_dir)