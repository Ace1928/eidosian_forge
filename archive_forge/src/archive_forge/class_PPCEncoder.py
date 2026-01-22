import struct
from typing import Union
class PPCEncoder(BCJFilter):

    def __init__(self):
        super().__init__(self.ppc_code, 4, True)