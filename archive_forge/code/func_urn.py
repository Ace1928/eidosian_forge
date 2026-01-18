import os
import sys
from enum import Enum, _simple_enum
@property
def urn(self):
    return 'urn:uuid:' + str(self)