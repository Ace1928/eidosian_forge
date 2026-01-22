from zope.interface import provider
from twisted.plugin import IPlugin
from twisted.test.test_plugin import ITestPlugin
@provider(ITestPlugin, IPlugin)
class FifthTestPlugin:
    """
    More documentation: I hate you.
    """

    @staticmethod
    def test1() -> None:
        pass