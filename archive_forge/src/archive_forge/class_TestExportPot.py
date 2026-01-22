import os
from breezy import ignores, osutils
from breezy.tests import TestCaseWithMemoryTransport
from breezy.tests.features import PluginLoadedFeature
class TestExportPot(TestCaseWithMemoryTransport):

    def test_export_pot(self):
        out, err = self.run_bzr('export-pot')
        self.assertContainsRe(err, 'Exporting messages from builtin command: add')
        self.assertContainsRe(out, 'help of \'change\' option\nmsgid "Select changes introduced by the specified revision.')

    def test_export_pot_plugin_unknown(self):
        out, err = self.run_bzr('export-pot --plugin=lalalala', retcode=3)
        self.assertContainsRe(err, 'ERROR: Plugin lalalala is not loaded')

    def test_export_pot_plugin(self):
        self.requireFeature(PluginLoadedFeature('launchpad'))
        out, err = self.run_bzr('export-pot --plugin=launchpad')
        self.assertContainsRe(err, 'Exporting messages from plugin command: launchpad-login in launchpad')
        self.assertContainsRe(out, 'msgid "Show or set the Launchpad user ID."')