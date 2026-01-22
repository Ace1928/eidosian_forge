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
def _addLocationLabel(self, parentElement: ET.Element, label: LocationLabelDescriptor) -> None:
    labelElement = ET.Element('label')
    labelElement.attrib['name'] = label.name
    if label.elidable:
        labelElement.attrib['elidable'] = 'true'
    if label.olderSibling:
        labelElement.attrib['oldersibling'] = 'true'
    self._addLabelNames(labelElement, label.labelNames)
    self._addLocationElement(labelElement, userLocation=label.userLocation)
    parentElement.append(labelElement)