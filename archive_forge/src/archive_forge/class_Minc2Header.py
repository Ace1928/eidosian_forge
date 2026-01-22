import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
class Minc2Header(MincHeader):

    @classmethod
    def may_contain_header(klass, binaryblock):
        return binaryblock[:4] == b'\x89HDF'