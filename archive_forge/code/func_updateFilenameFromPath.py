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
def updateFilenameFromPath(self, masters=True, instances=True, force=False):
    """Set a descriptor filename attr from the path and this document path.

        If the filename attribute is not None: skip it.
        """
    if masters:
        for descriptor in self.sources:
            if descriptor.filename is not None and (not force):
                continue
            if self.path is not None:
                descriptor.filename = self._posixRelativePath(descriptor.path)
    if instances:
        for descriptor in self.instances:
            if descriptor.filename is not None and (not force):
                continue
            if self.path is not None:
                descriptor.filename = self._posixRelativePath(descriptor.path)