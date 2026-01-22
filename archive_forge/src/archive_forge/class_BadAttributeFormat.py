import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadAttributeFormat(ArffException):
    """Error raised when some attribute declaration is in an invalid format."""
    message = 'Bad @ATTRIBUTE format, at line %d.'