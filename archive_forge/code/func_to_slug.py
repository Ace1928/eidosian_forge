import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def to_slug(value, incoming=None, errors='strict'):
    """Normalize string.

    Convert to lowercase, remove non-word characters, and convert spaces
    to hyphens.

    Inspired by Django's `slugify` filter.

    :param value: Text to slugify
    :param incoming: Text's current encoding
    :param errors: Errors handling policy. See here for valid
        values http://docs.python.org/2/library/codecs.html
    :returns: slugified unicode representation of `value`
    :raises TypeError: If text is not an instance of str
    """
    value = encodeutils.safe_decode(value, incoming, errors)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = SLUGIFY_STRIP_RE.sub('', value).strip().lower()
    return SLUGIFY_HYPHENATE_RE.sub('-', value)