import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def _shelve_creation(self, tree, file_id, from_transform, to_transform, kind, name, parent, version, existing_path=None):
    w_trans_id = from_transform.trans_id_file_id(file_id)
    if parent is not None and kind is not None:
        from_transform.delete_contents(w_trans_id)
    from_transform.unversion_file(w_trans_id)
    if existing_path is not None:
        s_trans_id = to_transform.trans_id_tree_path(existing_path)
    else:
        s_trans_id = to_transform.trans_id_file_id(file_id)
    if parent is not None:
        s_parent_id = to_transform.trans_id_file_id(parent)
        to_transform.adjust_path(name, s_parent_id, s_trans_id)
        if existing_path is None:
            if kind is None:
                to_transform.create_file([b''], s_trans_id)
            else:
                transform.create_from_tree(to_transform, s_trans_id, tree, tree.id2path(file_id))
    if version:
        to_transform.version_file(s_trans_id, file_id=file_id)