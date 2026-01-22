from zope.interface import provider
from twisted.plugin import IPlugin
from twisted.test.test_plugin import ITestPlugin, ITestPlugin2
@provider(ITestPlugin2, IPlugin)
class AnotherTestPlugin:
    """
    Another plugin used solely for testing purposes.
    """

    @staticmethod
    def test() -> None:
        pass