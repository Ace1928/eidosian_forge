from zope.interface import provider
from twisted.application.service import ServiceMaker
from twisted.plugin import IPlugin
from twisted.words import iwords
@provider(IPlugin, iwords.IProtocolPlugin)
class RelayChatInterface:
    name = 'irc'

    @classmethod
    def getFactory(cls, realm, portal):
        from twisted.words import service
        return service.IRCFactory(realm, portal)