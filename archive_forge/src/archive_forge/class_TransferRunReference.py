import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class TransferRunReference(Reference):
    _required_fields = frozenset(('transferRunName',))
    _format_str = '%(transferRunName)s'
    typename = 'transfer run'