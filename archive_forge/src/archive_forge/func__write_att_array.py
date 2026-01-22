import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _write_att_array(self, attributes):
    if attributes:
        self.fp.write(NC_ATTRIBUTE)
        self._pack_int(len(attributes))
        for name, values in attributes.items():
            self._pack_string(name)
            self._write_att_values(values)
    else:
        self.fp.write(ABSENT)