from ...state.storageprotos_pb2 import SenderKeyRecordStructure
from .senderkeystate import SenderKeyState
from ...invalidkeyidexception import InvalidKeyIdException

        :type id: int
        :type iteration: int
        :type chainKey: bytearray
        :type signatureKey: ECKeyPair
        