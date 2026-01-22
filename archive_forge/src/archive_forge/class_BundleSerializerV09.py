from ...testament import StrictTestament3
from ..bundle_data import BundleInfo
from . import _get_bundle_header
from .v08 import BundleReader, BundleSerializerV08
class BundleSerializerV09(BundleSerializerV08):
    """Serializer for bzr bundle format 0.9

    This format supports rich root data, for the nested-trees work, but also
    supports repositories that don't have rich root data.  It cannot be
    used to transfer from a knit2 repo into a knit1 repo, because that would
    be lossy.
    """

    def check_compatible(self):
        pass

    def _write_main_header(self):
        """Write the header for the changes"""
        f = self.to_file
        f.write(_get_bundle_header('0.9') + b'#\n')

    def _testament_sha1(self, revision_id):
        return StrictTestament3.from_revision(self.source, revision_id).as_sha1()

    def read(self, f):
        """Read the rest of the bundles from the supplied file.

        :param f: The file to read from
        :return: A list of bundles
        """
        return BundleReaderV09(f).info