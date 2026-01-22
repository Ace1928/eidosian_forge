from __future__ import annotations
import logging
import enum
from warnings import warn
from collections import OrderedDict
import fs
import fs.base
import fs.errors
import fs.osfs
import fs.path
from fontTools.misc.textTools import tobytes
from fontTools.misc import plistlib
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from fontTools.ufoLib.errors import GlifLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.validators import (
from fontTools.misc import etree
from fontTools.ufoLib import _UFOBaseIO, UFOFormatVersion
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
class _FetchComponentBasesParser(_BaseParser):

    def __init__(self):
        self.bases = []
        super().__init__()

    def startElementHandler(self, name, attrs):
        if name == 'component' and self._elementStack and (self._elementStack[-1] == 'outline'):
            base = attrs.get('base')
            if base is not None:
                self.bases.append(base)
        super().startElementHandler(name, attrs)

    def endElementHandler(self, name):
        if name == 'outline':
            raise _DoneParsing
        super().endElementHandler(name)