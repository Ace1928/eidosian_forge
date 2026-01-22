import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class _DataListMixin:
    """Mixin to return a list from decode_rows instead of a generator"""

    def decode_rows(self, stream, conversors):
        return list(super().decode_rows(stream, conversors))