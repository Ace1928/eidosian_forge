import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
Split values by commas and quotes according to api-wg

    :param value: value to be split

    .. versionadded:: 3.17
    