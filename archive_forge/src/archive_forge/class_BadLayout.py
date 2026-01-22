import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadLayout(ArffException):
    """Error raised when the layout of the ARFF file has something wrong."""
    message = 'Invalid layout of the ARFF file, at line %d.'

    def __init__(self, msg=''):
        super().__init__()
        if msg:
            self.message = BadLayout.message + ' ' + msg.replace('%', '%%')