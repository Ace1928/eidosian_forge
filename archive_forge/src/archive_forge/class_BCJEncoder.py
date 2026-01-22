import struct
from typing import Union
class BCJEncoder(BCJFilter):

    def __init__(self):
        super().__init__(self.x86_code, 5, True)