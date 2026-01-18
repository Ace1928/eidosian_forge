import time
from io import BytesIO
from dulwich import __version__ as dulwich_version
from dulwich.objects import Blob
from .. import __version__ as brz_version
from .. import branch as _mod_branch
from .. import diff as _mod_diff
from .. import errors, osutils
from .. import revision as _mod_revision
from ..merge_directive import BaseMergeDirective
from .mapping import object_mode
from .object_store import get_object_store
def to_files(self):
    return ((summary, patch.splitlines(True)) for summary, patch in self.patches)