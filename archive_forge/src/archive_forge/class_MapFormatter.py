import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
class MapFormatter(string.Formatter):
    """String formatting method to map string
    mapped to float data field
    used for sorting back to string."""

    def format_field(self, value, spec):
        if spec.endswith('h'):
            value = num2sym[int(value)]
            spec = spec[:-1] + 's'
        return super(MapFormatter, self).format_field(value, spec)