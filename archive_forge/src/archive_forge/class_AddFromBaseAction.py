import os
import sys
from . import errors, osutils, ui
from .i18n import gettext
class AddFromBaseAction(AddAction):
    """This class will try to extract file ids from another tree."""

    def __init__(self, base_tree, base_path, to_file=None, should_print=None):
        super().__init__(to_file=to_file, should_print=should_print)
        self.base_tree = base_tree
        self.base_path = base_path

    def __call__(self, inv, parent_ie, path, kind):
        file_id, base_path = self._get_base_file_id(path, parent_ie)
        if file_id is not None:
            if self.should_print:
                self._to_file.write('adding %s w/ file id from %s\n' % (path, base_path))
        else:
            file_id = super().__call__(inv, parent_ie, path, kind)
        return file_id

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