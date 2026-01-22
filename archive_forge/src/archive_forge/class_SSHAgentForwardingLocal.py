import os
from twisted.conch.ssh import agent, channel, keys
from twisted.internet import protocol, reactor
from twisted.logger import Logger
class SSHAgentForwardingLocal(protocol.Protocol):
    pass