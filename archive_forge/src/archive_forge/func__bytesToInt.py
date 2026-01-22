from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import ResultIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
from axolotl.state.prekeybundle import PreKeyBundle
from axolotl.identitykey import IdentityKey
from axolotl.ecc.curve import Curve
from axolotl.ecc.djbec import DjbECPublicKey
import binascii
import sys
@staticmethod
def _bytesToInt(val):
    if sys.version_info >= (3, 0):
        valEnc = val.encode('latin-1') if type(val) is str else val
    else:
        valEnc = val
    return int(binascii.hexlify(valEnc), 16)