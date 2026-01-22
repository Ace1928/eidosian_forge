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
class AxisMappingDescriptor(SimpleDescriptor):
    """Represents the axis mapping element: mapping an input location
    to an output location in the designspace.

    .. code:: python

        m1 = AxisMappingDescriptor()
        m1.inputLocation = {"weight": 900, "width": 150}
        m1.outputLocation = {"weight": 870}

    .. code:: xml

        <mappings>
            <mapping>
                <input>
                    <dimension name="weight" xvalue="900"/>
                    <dimension name="width" xvalue="150"/>
                </input>
                <output>
                    <dimension name="weight" xvalue="870"/>
                </output>
            </mapping>
        </mappings>
    """
    _attrs = ['inputLocation', 'outputLocation']

    def __init__(self, *, inputLocation=None, outputLocation=None, description=None, groupDescription=None):
        self.inputLocation: SimpleLocationDict = inputLocation or {}
        'dict. Axis values for the input of the mapping, in design space coordinates.\n\n        varLib.\n\n        .. versionadded:: 5.1\n        '
        self.outputLocation: SimpleLocationDict = outputLocation or {}
        'dict. Axis values for the output of the mapping, in design space coordinates.\n\n        varLib.\n\n        .. versionadded:: 5.1\n        '
        self.description = description
        'string. A description of the mapping.\n\n        varLib.\n\n        .. versionadded:: 5.2\n        '
        self.groupDescription = groupDescription
        'string. A description of the group of mappings.\n\n        varLib.\n\n        .. versionadded:: 5.2\n        '