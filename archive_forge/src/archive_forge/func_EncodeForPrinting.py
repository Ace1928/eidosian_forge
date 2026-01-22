import logging
import sys
from typing import Optional, TextIO
from absl import flags
from absl import logging as absl_logging
from googleapiclient import model
def EncodeForPrinting(o: object) -> str:
    """Safely encode an object as the encoding for sys.stdout."""
    encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
    if isinstance(o, type('')) and (not isinstance(o, str)):
        return o.encode(encoding, 'backslashreplace')
    else:
        return str(o)