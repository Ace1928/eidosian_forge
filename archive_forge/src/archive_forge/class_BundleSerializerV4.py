import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
class BundleSerializerV4(bundle_serializer.BundleSerializer):
    """Implement the high-level bundle interface"""

    def write_bundle(self, repository, target, base, fileobj):
        """Write a bundle to a file object

        :param repository: The repository to retrieve revision data from
        :param target: The head revision to include ancestors of
        :param base: The ancestor of the target to stop including acestors
            at.
        :param fileobj: The file-like object to write to
        """
        write_op = BundleWriteOperation(base, target, repository, fileobj)
        return write_op.do_write()

    def read(self, file):
        """return a reader object for a given file"""
        bundle = BundleInfoV4(file, self)
        return bundle

    @staticmethod
    def get_source_serializer(info):
        """Retrieve the serializer for a given info object"""
        return serializer.format_registry.get(info[b'serializer'].decode('ascii'))