import struct
from typing import Union
class ARMDecoder(BCJFilter):

    def __init__(self, size: int):
        super().__init__(self.arm_code, 4, False, size)