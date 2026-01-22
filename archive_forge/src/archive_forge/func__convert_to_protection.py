from __future__ import annotations
import mmap
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import (
from pandas.io.excel._util import (
@classmethod
def _convert_to_protection(cls, protection_dict):
    """
        Convert ``protection_dict`` to an openpyxl v2 Protection object.

        Parameters
        ----------
        protection_dict : dict
            A dict with zero or more of the following keys.
                'locked'
                'hidden'

        Returns
        -------
        """
    from openpyxl.styles import Protection
    return Protection(**protection_dict)