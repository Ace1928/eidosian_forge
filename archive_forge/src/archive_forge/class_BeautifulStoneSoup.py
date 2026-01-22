from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
class BeautifulStoneSoup(BeautifulSoup):
    """Deprecated interface to an XML parser."""

    def __init__(self, *args, **kwargs):
        kwargs['features'] = 'xml'
        warnings.warn('The BeautifulStoneSoup class is deprecated. Instead of using it, pass features="xml" into the BeautifulSoup constructor.', DeprecationWarning, stacklevel=2)
        super(BeautifulStoneSoup, self).__init__(*args, **kwargs)