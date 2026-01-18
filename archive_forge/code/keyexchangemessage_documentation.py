from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from . import whisperprotos_pb2 as whisperprotos
from ..legacymessageexception import LegacyMessageException
from ..invalidversionexception import InvalidVersionException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
from ..ecc.curve import Curve
from ..identitykey import IdentityKey

        :type messageVersion: int
        :type  sequence: int
        :type flags:int
        :type baseKey: ECPublicKey
        :type baseKeySignature: bytearray
        :type ratchetKey: ECPublicKey
        :type identityKey: IdentityKey
        :type serialized: bytearray
        