import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class EncodedNominalConversor:

    def __init__(self, values):
        self.values = {v: i for i, v in enumerate(values)}
        self.values[0] = 0

    def __call__(self, value):
        try:
            return self.values[value]
        except KeyError:
            raise BadNominalValue(value)