import os
import sys
from . import errors, osutils, ui
from .i18n import gettext
def _get_base_file_id(self, path, parent_ie):
    """Look for a file id in the base branch.

        First, if the base tree has the parent directory,
        we look for a file with the same name in that directory.
        Else, we look for an entry in the base tree with the same path.
        """
    try:
        parent_path = self.base_tree.id2path(parent_ie.file_id)
    except errors.NoSuchId:
        pass
    else:
        base_path = osutils.pathjoin(parent_path, osutils.basename(path))
        base_id = self.base_tree.path2id(base_path)
        if base_id is not None:
            return (base_id, base_path)
    full_base_path = osutils.pathjoin(self.base_path, path)
    return (self.base_tree.path2id(full_base_path), full_base_path)