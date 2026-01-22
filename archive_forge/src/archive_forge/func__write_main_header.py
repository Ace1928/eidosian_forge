from ...testament import StrictTestament3
from ..bundle_data import BundleInfo
from . import _get_bundle_header
from .v08 import BundleReader, BundleSerializerV08
def _write_main_header(self):
    """Write the header for the changes"""
    f = self.to_file
    f.write(_get_bundle_header('0.9') + b'#\n')