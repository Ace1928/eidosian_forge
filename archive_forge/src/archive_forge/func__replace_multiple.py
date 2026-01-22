from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def _replace_multiple(value, needles_and_replacements):

    def replacer(match):
        return needles_and_replacements[match.group(0)]
    pattern = re.compile('|'.join([re.escape(needle) for needle in needles_and_replacements.keys()]))
    result = pattern.sub(replacer, value)
    return result