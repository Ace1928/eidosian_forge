import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def adjustWindow(self, channel, bytesToAdd):
    """
        Tell the other side that we will receive more data.  This should not
        normally need to be called as it is managed automatically.

        @type channel:      subclass of L{SSHChannel}
        @type bytesToAdd:   L{int}
        """
    if channel.localClosed:
        return
    packet = struct.pack('>2L', self.channelsToRemoteChannel[channel], bytesToAdd)
    self.transport.sendPacket(MSG_CHANNEL_WINDOW_ADJUST, packet)
    self._log.debug('adding {bytesToAdd} to {localWindowLeft} in channel {id}', bytesToAdd=bytesToAdd, localWindowLeft=channel.localWindowLeft, id=channel.id)
    channel.localWindowLeft += bytesToAdd