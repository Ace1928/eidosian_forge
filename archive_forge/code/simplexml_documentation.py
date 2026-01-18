from __future__ import unicode_literals
import sys
import logging
import re
import time
import xml.dom.minidom
from . import __author__, __copyright__, __license__, __version__
from .helpers import TYPE_MAP, TYPE_MARSHAL_FN, TYPE_UNMARSHAL_FN, \
Return the XML representation of this tag