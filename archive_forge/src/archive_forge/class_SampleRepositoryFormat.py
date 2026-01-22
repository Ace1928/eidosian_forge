from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class SampleRepositoryFormat(bzrrepository.RepositoryFormatMetaDir):
    """A sample format

    this format is initializable, unsupported to aid in testing the
    open and open(unsupported=True) routines.
    """

    @classmethod
    def get_format_string(cls):
        """See RepositoryFormat.get_format_string()."""
        return b'Sample .bzr repository format.'

    def initialize(self, a_controldir, shared=False):
        """Initialize a repository in a BzrDir"""
        t = a_controldir.get_repository_transport(self)
        t.put_bytes('format', self.get_format_string())
        return 'A bzr repository dir'

    def is_supported(self):
        return False

    def open(self, a_controldir, _found=False):
        return 'opened repository.'