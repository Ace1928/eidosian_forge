from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class TestBzrBranchFormat(tests.TestCaseWithTransport):
    """Tests for the BzrBranchFormat facility."""

    def test_find_format(self):
        self.build_tree(['foo/', 'bar/'])

        def check_format(format, url):
            dir = format._matchingcontroldir.initialize(url)
            dir.create_repository()
            format.initialize(dir)
            found_format = _mod_bzrbranch.BranchFormatMetadir.find_format(dir)
            self.assertIsInstance(found_format, format.__class__)
        check_format(BzrBranchFormat5(), 'bar')

    def test_from_string(self):
        self.assertIsInstance(SampleBranchFormat.from_string(b'Sample branch format.'), SampleBranchFormat)
        self.assertRaises(AssertionError, SampleBranchFormat.from_string, b'Different branch format.')

    def test_find_format_not_branch(self):
        dir = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        self.assertRaises(errors.NotBranchError, _mod_bzrbranch.BranchFormatMetadir.find_format, dir)

    def test_find_format_unknown_format(self):
        dir = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        SampleBranchFormat().initialize(dir)
        self.assertRaises(errors.UnknownFormatError, _mod_bzrbranch.BranchFormatMetadir.find_format, dir)

    def test_find_format_with_features(self):
        tree = self.make_branch_and_tree('.', format='2a')
        tree.branch.update_feature_flags({b'name': b'optional'})
        found_format = _mod_bzrbranch.BranchFormatMetadir.find_format(tree.controldir)
        self.assertIsInstance(found_format, _mod_bzrbranch.BranchFormatMetadir)
        self.assertEqual(found_format.features.get(b'name'), b'optional')
        tree.branch.update_feature_flags({b'name': None})
        branch = _mod_branch.Branch.open('.')
        self.assertEqual(branch._format.features, {})