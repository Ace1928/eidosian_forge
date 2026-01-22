from __future__ import absolute_import, division, unicode_literals
from six import text_type
import re
from copy import copy
from . import base
from .. import _ihatexml
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def _setChildNodes(self, value):
    del self._element[:]
    self._childNodes = []
    for element in value:
        self.insertChild(element)