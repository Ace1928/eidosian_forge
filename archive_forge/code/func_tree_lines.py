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
def tree_lines(tree, path, require_text=False):
    try:
        tree_file = tree.get_file(path)
    except _mod_transport.NoSuchFile:
        return []
    else:
        if require_text is True:
            tree_file = text_file(tree_file)
        return tree_file.readlines()