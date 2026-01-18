from twisted.plugin import getPlugins
from twisted.trial import unittest
from twisted.trial.itrial import IReporter
def test_subunitPlugin(self) -> None:
    """
        One of the reporter plugins is the subunit reporter plugin.
        """
    subunitPlugin = self.getPluginsByLongOption('subunit')
    self.assertEqual('Subunit Reporter', subunitPlugin.name)
    self.assertEqual('twisted.trial.reporter', subunitPlugin.module)
    self.assertEqual('subunit', subunitPlugin.longOpt)
    self.assertIdentical(None, subunitPlugin.shortOpt)
    self.assertEqual('SubunitReporter', subunitPlugin.klass)