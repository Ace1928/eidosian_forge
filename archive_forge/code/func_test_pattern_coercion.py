import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings
from humanfriendly import (
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
from humanfriendly.sphinx import (
from humanfriendly.tables import (
from humanfriendly.terminal import (
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
from humanfriendly.text import (
from humanfriendly.usage import (
from mock import MagicMock
def test_pattern_coercion(self):
    """Test :func:`humanfriendly.coerce_pattern()`."""
    empty_pattern = re.compile('')
    assert isinstance(coerce_pattern('foobar'), type(empty_pattern))
    assert empty_pattern is coerce_pattern(empty_pattern)
    pattern = coerce_pattern('foobar', re.IGNORECASE)
    assert pattern.match('FOOBAR')
    with self.assertRaises(ValueError):
        coerce_pattern([])