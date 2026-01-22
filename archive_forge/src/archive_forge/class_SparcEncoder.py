import struct
from typing import Union
class SparcEncoder(BCJFilter):

    def __init__(self):
        super().__init__(self.sparc_code, 4, True)