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
def _addAxisLabel(self, axisElement: ET.Element, label: AxisLabelDescriptor) -> None:
    labelElement = ET.Element('label')
    labelElement.attrib['uservalue'] = self.intOrFloat(label.userValue)
    if label.userMinimum is not None:
        labelElement.attrib['userminimum'] = self.intOrFloat(label.userMinimum)
    if label.userMaximum is not None:
        labelElement.attrib['usermaximum'] = self.intOrFloat(label.userMaximum)
    labelElement.attrib['name'] = label.name
    if label.elidable:
        labelElement.attrib['elidable'] = 'true'
    if label.olderSibling:
        labelElement.attrib['oldersibling'] = 'true'
    if label.linkedUserValue is not None:
        labelElement.attrib['linkeduservalue'] = self.intOrFloat(label.linkedUserValue)
    self._addLabelNames(labelElement, label.labelNames)
    axisElement.append(labelElement)