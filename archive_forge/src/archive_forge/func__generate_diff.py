import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
@staticmethod
def _generate_diff(repository, revision_id, ancestor_id):
    tree_1 = repository.revision_tree(ancestor_id)
    tree_2 = repository.revision_tree(revision_id)
    s = BytesIO()
    diff.show_diff_trees(tree_1, tree_2, s, old_label='', new_label='')
    return s.getvalue()