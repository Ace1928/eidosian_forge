import hmac
import hashlib
from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..ecc.curve import Curve
from . import whisperprotos_pb2 as whisperprotos
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
def verifyMac(self, messageVersion, senderIdentityKey, receiverIdentityKey, macKey):
    parts = ByteUtil.split(self.serialized, len(self.serialized) - self.__class__.MAC_LENGTH, self.__class__.MAC_LENGTH)
    ourMac = self.getMac(messageVersion, senderIdentityKey, receiverIdentityKey, macKey, parts[0])
    theirMac = parts[1]
    if ourMac != theirMac:
        raise InvalidMessageException('Bad Mac!')