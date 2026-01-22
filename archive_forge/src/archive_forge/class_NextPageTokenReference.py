import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class NextPageTokenReference(Reference):
    _required_fields = frozenset(('pageTokenId',))
    _format_str = '%(pageTokenId)s'
    typename = 'page token'