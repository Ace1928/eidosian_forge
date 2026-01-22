from __future__ import unicode_literals
import re
from xml.sax.saxutils import unescape
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.constants import namespaces
from tensorboard._vendor.html5lib.filters import sanitizer
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def _attr_filter(tag, attr, value):
    return attr in attributes