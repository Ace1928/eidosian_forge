import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def write_modified(self, results):
    if not self.working_tree.supports_merge_modified():
        return
    modified_hashes = {}
    for path in results.modified_paths:
        wt_relpath = self.working_tree.relpath(path)
        if not self.working_tree.is_versioned(wt_relpath):
            continue
        hash = self.working_tree.get_file_sha1(wt_relpath)
        if hash is None:
            continue
        modified_hashes[wt_relpath] = hash
    self.working_tree.set_merge_modified(modified_hashes)