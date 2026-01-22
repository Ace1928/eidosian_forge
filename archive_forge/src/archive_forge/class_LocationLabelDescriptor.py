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
class LocationLabelDescriptor(SimpleDescriptor):
    """Container for location label data.

    Analogue of OpenType's STAT data for a free-floating location (format 4).
    All values are user values.

    See: `OTSpec STAT Axis value table, format 4 <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-value-table-format-4>`_

    .. versionadded:: 5.0
    """
    flavor = 'label'
    _attrs = ('name', 'elidable', 'olderSibling', 'userLocation', 'labelNames')

    def __init__(self, *, name, userLocation, elidable=False, olderSibling=False, labelNames=None):
        self.name: str = name
        'Label for this named location, STAT field ``valueNameID``.'
        self.userLocation: SimpleLocationDict = userLocation or {}
        'Location in user coordinates along each axis.\n\n        If an axis is not mentioned, it is assumed to be at its default location.\n\n        .. seealso:: This may be only part of the full location. See:\n           :meth:`getFullUserLocation`\n        '
        self.elidable: bool = elidable
        'STAT flag ``ELIDABLE_AXIS_VALUE_NAME``.\n\n        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_\n        '
        self.olderSibling: bool = olderSibling
        'STAT flag ``OLDER_SIBLING_FONT_ATTRIBUTE``.\n\n        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_\n        '
        self.labelNames: Dict[str, str] = labelNames or {}
        "User-facing translations of this location's label. Keyed by\n        xml:lang code.\n        "

    @property
    def defaultName(self) -> str:
        """Return the English name from :attr:`labelNames` or the :attr:`name`."""
        return self.labelNames.get('en') or self.name

    def getFullUserLocation(self, doc: 'DesignSpaceDocument') -> SimpleLocationDict:
        """Get the complete user location of this label, by combining data
        from the explicit user location and default axis values.

        .. versionadded:: 5.0
        """
        return {axis.name: self.userLocation.get(axis.name, axis.default) for axis in doc.axes}