from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
class FeatureNotFound(ValueError):
    """Exception raised by the BeautifulSoup constructor if no parser with the
    requested features is found.
    """
    pass