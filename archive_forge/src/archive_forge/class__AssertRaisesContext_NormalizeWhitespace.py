import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
class _AssertRaisesContext_NormalizeWhitespace(_unittest.case._AssertRaisesContext):

    def __exit__(self, exc_type, exc_value, tb):
        try:
            _save_re = self.expected_regex
            self.expected_regex = None
            if not super().__exit__(exc_type, exc_value, tb):
                return False
        finally:
            self.expected_regex = _save_re
        exc_value = re.sub('(?s)\\s+', ' ', str(exc_value))
        if not _save_re.search(exc_value):
            self._raiseFailure('"{}" does not match "{}"'.format(_save_re.pattern, exc_value))
        return True