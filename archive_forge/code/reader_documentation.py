import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError

         Read output file line by line. When the `line` matches the pattern
        of certain keywords in `param.[dtype]_keys`, for example,

        if line in param.string_keys:
            out_data[key] = read_string(line)

        parse that line and store it to `out_data` in specified data type.
         To cover all `dtype` parameters, for loop was used,

        for [dtype] in parameters_keys:
            if line in param.[dtype]_keys:
                out_data[key] = read_[dtype](line)

        After found matched pattern, escape the for loop using `continue`.
        