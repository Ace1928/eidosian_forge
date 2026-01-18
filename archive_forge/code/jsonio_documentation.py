import datetime
import json
import numpy as np
from ase.utils import reader, writer
Convert "int" keys: "1" -> 1.

    The json.dump() function will convert int keys in dicts to str keys.
    This function goes the other way.
    