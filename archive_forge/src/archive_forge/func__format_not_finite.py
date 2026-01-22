from __future__ import annotations
import math
import re
import sys
from fractions import Fraction
from typing import TYPE_CHECKING
from .i18n import _gettext as _
from .i18n import _ngettext, decimal_separator, thousands_separator
from .i18n import _ngettext_noop as NS_
from .i18n import _pgettext as P_
def _format_not_finite(value: float) -> str:
    """Utility function to handle infinite and nan cases."""
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value) and value < 0:
        return '-Inf'
    if math.isinf(value) and value > 0:
        return '+Inf'
    return ''