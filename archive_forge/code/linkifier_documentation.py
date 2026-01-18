from __future__ import unicode_literals
import re
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
Handle the "a" tag

        This could adjust the link or drop it altogether depending on what the
        callbacks return.

        This yields the new set of tokens.

        