import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
@staticmethod
def _content_from_tree(tt, tree, file_id):
    trans_id = tt.trans_id_file_id(file_id)
    tt.delete_contents(trans_id)
    transform.create_from_tree(tt, trans_id, tree, tree.id2path(file_id))