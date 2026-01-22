from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
class DBus(MessageGenerator):
    """Messages to talk to the message bus
    """
    interface = 'org.freedesktop.DBus'

    def __init__(self, object_path='/org/freedesktop/DBus', bus_name='org.freedesktop.DBus'):
        super().__init__(object_path=object_path, bus_name=bus_name)

    def Hello(self):
        return new_method_call(self, 'Hello')

    def RequestName(self, name, flags=0):
        return new_method_call(self, 'RequestName', 'su', (name, flags))

    def ReleaseName(self, name):
        return new_method_call(self, 'ReleaseName', 's', (name,))

    def StartServiceByName(self, name):
        return new_method_call(self, 'StartServiceByName', 'su', (name, 0))

    def UpdateActivationEnvironment(self, env):
        return new_method_call(self, 'UpdateActivationEnvironment', 'a{ss}', (env,))

    def NameHasOwner(self, name):
        return new_method_call(self, 'NameHasOwner', 's', (name,))

    def ListNames(self):
        return new_method_call(self, 'ListNames')

    def ListActivatableNames(self):
        return new_method_call(self, 'ListActivatableNames')

    def AddMatch(self, rule):
        """*rule* can be a str or a :class:`MatchRule` instance"""
        if isinstance(rule, MatchRule):
            rule = rule.serialise()
        return new_method_call(self, 'AddMatch', 's', (rule,))

    def RemoveMatch(self, rule):
        if isinstance(rule, MatchRule):
            rule = rule.serialise()
        return new_method_call(self, 'RemoveMatch', 's', (rule,))

    def GetNameOwner(self, name):
        return new_method_call(self, 'GetNameOwner', 's', (name,))

    def ListQueuedOwners(self, name):
        return new_method_call(self, 'ListQueuedOwners', 's', (name,))

    def GetConnectionUnixUser(self, name):
        return new_method_call(self, 'GetConnectionUnixUser', 's', (name,))

    def GetConnectionUnixProcessID(self, name):
        return new_method_call(self, 'GetConnectionUnixProcessID', 's', (name,))

    def GetAdtAuditSessionData(self, name):
        return new_method_call(self, 'GetAdtAuditSessionData', 's', (name,))

    def GetConnectionSELinuxSecurityContext(self, name):
        return new_method_call(self, 'GetConnectionSELinuxSecurityContext', 's', (name,))

    def ReloadConfig(self):
        return new_method_call(self, 'ReloadConfig')

    def GetId(self):
        return new_method_call(self, 'GetId')

    def GetConnectionCredentials(self, name):
        return new_method_call(self, 'GetConnectionCredentials', 's', (name,))