from __future__ import absolute_import, division, unicode_literals
from six import text_type
import re
from copy import copy
from . import base
from .. import _ihatexml
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def _getETreeTag(self, name, namespace):
    if namespace is None:
        etree_tag = name
    else:
        etree_tag = '{%s}%s' % (namespace, name)
    return etree_tag