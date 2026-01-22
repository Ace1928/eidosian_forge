from .... import errors
from .... import transport as _mod_transport
from .... import ui
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....textfile import text_file
from ....timestamp import format_highres_date
from ....trace import mutter
from ...testament import StrictTestament
from ..bundle_data import BundleInfo, RevisionInfo
from . import BundleSerializer, _get_bundle_header, binary_diff
def _read_patches(self):
    do_continue = True
    revision_actions = []
    while do_continue:
        action, lines, do_continue = self._read_one_patch()
        if action is not None:
            revision_actions.append((action, lines))
    if self.info.revisions[-1].tree_actions is not None:
        raise AssertionError()
    self.info.revisions[-1].tree_actions = revision_actions