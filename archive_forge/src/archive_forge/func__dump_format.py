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
def _dump_format(self):
    for key in self:
        value = self.get_as_string(key)
        if not value or value[0] == '\n':
            entry = '%s:%s\n' % (key, value)
        else:
            entry = '%s: %s\n' % (key, value)
        yield entry