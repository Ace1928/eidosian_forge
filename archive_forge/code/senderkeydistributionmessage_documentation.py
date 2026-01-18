from .ciphertextmessage import CiphertextMessage
from . import whisperprotos_pb2 as whisperprotos
from ..util.byteutil import ByteUtil
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
from ..ecc.curve import Curve

        :type id: int
        :type iteration: int
        :type chainKey: bytearray
        :type signatureKey: ECPublicKey
        