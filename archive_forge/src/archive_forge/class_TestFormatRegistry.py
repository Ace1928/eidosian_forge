import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
class TestFormatRegistry(TestCase):

    def make_format_registry(self):
        my_format_registry = controldir.ControlDirFormatRegistry()
        my_format_registry.register('deprecated', DeprecatedBzrDirFormat, 'Some format.  Slower and unawesome and deprecated.', deprecated=True)
        my_format_registry.register_lazy('lazy', __name__, 'DeprecatedBzrDirFormat', 'Format registered lazily', deprecated=True)
        bzr.register_metadir(my_format_registry, 'knit', 'breezy.bzr.knitrepo.RepositoryFormatKnit1', 'Format using knits')
        my_format_registry.set_default('knit')
        bzr.register_metadir(my_format_registry, 'branch6', 'breezy.bzr.knitrepo.RepositoryFormatKnit3', 'Experimental successor to knit.  Use at your own risk.', branch_format='breezy.bzr.branch.BzrBranchFormat6', experimental=True)
        bzr.register_metadir(my_format_registry, 'hidden format', 'breezy.bzr.knitrepo.RepositoryFormatKnit3', 'Experimental successor to knit.  Use at your own risk.', branch_format='breezy.bzr.branch.BzrBranchFormat6', hidden=True)
        my_format_registry.register('hiddendeprecated', DeprecatedBzrDirFormat, 'Old format.  Slower and does not support things. ', hidden=True)
        my_format_registry.register_lazy('hiddenlazy', __name__, 'DeprecatedBzrDirFormat', 'Format registered lazily', deprecated=True, hidden=True)
        return my_format_registry

    def test_format_registry(self):
        my_format_registry = self.make_format_registry()
        my_bzrdir = my_format_registry.make_controldir('lazy')
        self.assertIsInstance(my_bzrdir, DeprecatedBzrDirFormat)
        my_bzrdir = my_format_registry.make_controldir('deprecated')
        self.assertIsInstance(my_bzrdir, DeprecatedBzrDirFormat)
        my_bzrdir = my_format_registry.make_controldir('default')
        self.assertIsInstance(my_bzrdir.repository_format, knitrepo.RepositoryFormatKnit1)
        my_bzrdir = my_format_registry.make_controldir('knit')
        self.assertIsInstance(my_bzrdir.repository_format, knitrepo.RepositoryFormatKnit1)
        my_bzrdir = my_format_registry.make_controldir('branch6')
        self.assertIsInstance(my_bzrdir.get_branch_format(), breezy.bzr.branch.BzrBranchFormat6)

    def test_get_help(self):
        my_format_registry = self.make_format_registry()
        self.assertEqual('Format registered lazily', my_format_registry.get_help('lazy'))
        self.assertEqual('Format using knits', my_format_registry.get_help('knit'))
        self.assertEqual('Format using knits', my_format_registry.get_help('default'))
        self.assertEqual('Some format.  Slower and unawesome and deprecated.', my_format_registry.get_help('deprecated'))

    def test_help_topic(self):
        topics = help_topics.HelpTopicRegistry()
        registry = self.make_format_registry()
        topics.register('current-formats', registry.help_topic, 'Current formats')
        topics.register('other-formats', registry.help_topic, 'Other formats')
        new = topics.get_detail('current-formats')
        rest = topics.get_detail('other-formats')
        experimental, deprecated = rest.split('Deprecated formats')
        self.assertContainsRe(new, 'formats-help')
        self.assertContainsRe(new, ':knit:\n    \\(native\\) \\(default\\) Format using knits\n')
        self.assertContainsRe(experimental, ':branch6:\n    \\(native\\) Experimental successor to knit')
        self.assertContainsRe(deprecated, ':lazy:\n    \\(native\\) Format registered lazily\n')
        self.assertNotContainsRe(new, 'hidden')

    def test_set_default_repository(self):
        default_factory = controldir.format_registry.get('default')
        old_default = [k for k, v in controldir.format_registry.iteritems() if v == default_factory and k != 'default'][0]
        controldir.format_registry.set_default_repository('dirstate-with-subtree')
        try:
            self.assertIs(controldir.format_registry.get('dirstate-with-subtree'), controldir.format_registry.get('default'))
            self.assertIs(repository.format_registry.get_default().__class__, knitrepo.RepositoryFormatKnit3)
        finally:
            controldir.format_registry.set_default_repository(old_default)

    def test_aliases(self):
        a_registry = controldir.ControlDirFormatRegistry()
        a_registry.register('deprecated', DeprecatedBzrDirFormat, 'Old format.  Slower and does not support stuff', deprecated=True)
        a_registry.register_alias('deprecatedalias', 'deprecated')
        self.assertEqual({'deprecatedalias': 'deprecated'}, a_registry.aliases())