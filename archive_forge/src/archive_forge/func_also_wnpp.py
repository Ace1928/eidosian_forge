import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@property
def also_wnpp(self):
    """ list of WNPP bug numbers closed by the removal

        The bug numbers are returned as integers.
        """
    if 'also-wnpp' not in self:
        return []
    return [int(b) for b in self['also-wnpp'].split(' ')]