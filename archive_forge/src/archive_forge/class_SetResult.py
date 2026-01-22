import enum
import threading
from cupyx.distributed import _klv_utils
class SetResult:

    def klv(self):
        v = bytearray(bytes(True))
        action = _klv_utils.get_result_action_t(0, v)
        return bytes(action)

    @staticmethod
    def from_klv(klv):
        return True