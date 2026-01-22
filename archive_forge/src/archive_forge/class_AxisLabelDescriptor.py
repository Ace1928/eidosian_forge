from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
class AxisLabelDescriptor(SimpleDescriptor):
    """Container for axis label data.

    Analogue of OpenType's STAT data for a single axis (formats 1, 2 and 3).
    All values are user values.
    See: `OTSpec STAT Axis value table, format 1, 2, 3 <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-value-table-format-1>`_

    The STAT format of the Axis value depends on which field are filled-in,
    see :meth:`getFormat`

    .. versionadded:: 5.0
    """
    flavor = 'label'
    _attrs = ('userMinimum', 'userValue', 'userMaximum', 'name', 'elidable', 'olderSibling', 'linkedUserValue', 'labelNames')

    def __init__(self, *, name, userValue, userMinimum=None, userMaximum=None, elidable=False, olderSibling=False, linkedUserValue=None, labelNames=None):
        self.userMinimum: Optional[float] = userMinimum
        'STAT field ``rangeMinValue`` (format 2).'
        self.userValue: float = userValue
        'STAT field ``value`` (format 1, 3) or ``nominalValue`` (format 2).'
        self.userMaximum: Optional[float] = userMaximum
        'STAT field ``rangeMaxValue`` (format 2).'
        self.name: str = name
        'Label for this axis location, STAT field ``valueNameID``.'
        self.elidable: bool = elidable
        'STAT flag ``ELIDABLE_AXIS_VALUE_NAME``.\n\n        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_\n        '
        self.olderSibling: bool = olderSibling
        'STAT flag ``OLDER_SIBLING_FONT_ATTRIBUTE``.\n\n        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_\n        '
        self.linkedUserValue: Optional[float] = linkedUserValue
        'STAT field ``linkedValue`` (format 3).'
        self.labelNames: MutableMapping[str, str] = labelNames or {}
        "User-facing translations of this location's label. Keyed by\n        ``xml:lang`` code.\n        "

    def getFormat(self) -> int:
        """Determine which format of STAT Axis value to use to encode this label.

        ===========  =========  ===========  ===========  ===============
        STAT Format  userValue  userMinimum  userMaximum  linkedUserValue
        ===========  =========  ===========  ===========  ===============
        1            ✅          ❌            ❌            ❌
        2            ✅          ✅            ✅            ❌
        3            ✅          ❌            ❌            ✅
        ===========  =========  ===========  ===========  ===============
        """
        if self.linkedUserValue is not None:
            return 3
        if self.userMinimum is not None or self.userMaximum is not None:
            return 2
        return 1

    @property
    def defaultName(self) -> str:
        """Return the English name from :attr:`labelNames` or the :attr:`name`."""
        return self.labelNames.get('en') or self.name