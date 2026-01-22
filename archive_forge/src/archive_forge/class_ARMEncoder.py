import struct
from typing import Union
class ARMEncoder(BCJFilter):

    def __init__(self):
        super().__init__(self.arm_code, 4, True)