from collections.abc import Iterable
from io import BytesIO
import os
import re
import shutil
import sys
import tempfile
from unittest import TestCase as _TestCase
from fontTools.config import Config
from fontTools.misc.textTools import tobytes
from fontTools.misc.xmlWriter import XMLWriter
class AllocatingDict(dict):

    def __missing__(reverseDict, key):
        self._glyphOrder.append(key)
        gid = len(reverseDict)
        reverseDict[key] = gid
        return gid