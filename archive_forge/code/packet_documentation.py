import inspect
import struct
import base64
from . import packet_base
from . import ethernet
from os_ken import utils
from os_ken.lib.stringify import StringifyMixin
Returns the firstly found protocol that matches to the
        specified protocol.
        