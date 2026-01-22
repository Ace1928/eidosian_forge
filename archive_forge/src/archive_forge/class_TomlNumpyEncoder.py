import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
class TomlNumpyEncoder(TomlEncoder):

    def __init__(self, _dict=dict, preserve=False):
        import numpy as np
        super(TomlNumpyEncoder, self).__init__(_dict, preserve)
        self.dump_funcs[np.float16] = _dump_float
        self.dump_funcs[np.float32] = _dump_float
        self.dump_funcs[np.float64] = _dump_float
        self.dump_funcs[np.int16] = self._dump_int
        self.dump_funcs[np.int32] = self._dump_int
        self.dump_funcs[np.int64] = self._dump_int

    def _dump_int(self, v):
        return '{}'.format(int(v))