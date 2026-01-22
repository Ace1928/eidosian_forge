import hashlib
import socket
import time
from pyu2f import errors
from pyu2f import hardware
from pyu2f import hidtransport
from pyu2f import model
def GetLocalU2FInterface(origin=socket.gethostname()):
    """Obtains a U2FInterface for the first valid local U2FHID device found."""
    hid_transports = hidtransport.DiscoverLocalHIDU2FDevices()
    for t in hid_transports:
        try:
            return U2FInterface(security_key=hardware.SecurityKey(transport=t), origin=origin)
        except errors.UnsupportedVersionException:
            pass
    raise errors.NoDeviceFoundError()