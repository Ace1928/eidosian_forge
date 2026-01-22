from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
def _check_tag(tag):
    _check_element('tag', tag, _TAG_CHARS, 1, 127)