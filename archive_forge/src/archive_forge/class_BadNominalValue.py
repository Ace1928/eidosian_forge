import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadNominalValue(ArffException):
    """Error raised when a value in used in some data instance but is not
    declared into it respective attribute declaration."""

    def __init__(self, value):
        super().__init__()
        self.message = 'Data value %s not found in nominal declaration, ' % value + 'at line %d.'