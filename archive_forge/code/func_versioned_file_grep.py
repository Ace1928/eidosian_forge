import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def versioned_file_grep(tree, tree_path, relpath, path, opts, revno, path_prefix=None):
    """Create a file object for the specified id and pass it on to _file_grep.
    """
    path = _make_display_path(relpath, path)
    file_text = tree.get_file_text(tree_path)
    _file_grep(file_text, path, opts, revno, path_prefix)