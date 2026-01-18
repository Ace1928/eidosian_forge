import base64
import struct
from os_ken.lib import addrconv
@staticmethod
def to_user(data):
    return base64.b64encode(data).decode('ascii')