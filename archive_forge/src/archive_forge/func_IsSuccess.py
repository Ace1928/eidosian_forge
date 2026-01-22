import struct
from pyu2f import errors
def IsSuccess(self):
    return self.sw1 == 144 and self.sw2 == 0