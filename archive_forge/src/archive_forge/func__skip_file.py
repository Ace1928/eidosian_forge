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
def _skip_file(include, exclude, path):
    if include and (not _path_in_glob_list(path, include)):
        return True
    if exclude and _path_in_glob_list(path, exclude):
        return True
    return False