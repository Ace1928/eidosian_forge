import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
@staticmethod
def _decode_values(values, conversors):
    try:
        values = [None if value is None else conversor(value) for conversor, value in zip(conversors, values)]
    except ValueError as exc:
        if 'float: ' in str(exc):
            raise BadNumericalValue()
    return values