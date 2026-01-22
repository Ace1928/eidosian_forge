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
def _update_tree(self, bundle_tree, revision_id):
    bundle_tree.note_last_changed('', revision_id)
    BundleInfo._update_tree(self, bundle_tree, revision_id)