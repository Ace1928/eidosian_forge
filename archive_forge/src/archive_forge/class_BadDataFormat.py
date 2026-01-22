import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadDataFormat(ArffException):
    """Error raised when some data instance is in an invalid format."""

    def __init__(self, value):
        super().__init__()
        self.message = 'Bad @DATA instance format in line %d: ' + '%s' % value