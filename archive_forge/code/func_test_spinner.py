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
def test_spinner(self):
    """Test :func:`humanfriendly.Spinner`."""
    stream = StringIO()
    spinner = Spinner(label='test spinner', total=4, stream=stream, interactive=True)
    for progress in [1, 2, 3, 4]:
        spinner.step(progress=progress)
        time.sleep(0.2)
    spinner.clear()
    output = stream.getvalue()
    output = output.replace(ANSI_SHOW_CURSOR, '').replace(ANSI_HIDE_CURSOR, '')
    lines = [line for line in output.split(ANSI_ERASE_LINE) if line]
    self.assertTrue(len(lines) > 0)
    self.assertTrue(all(('test spinner' in line for line in lines)))
    self.assertTrue(all(('%' in line for line in lines)))
    self.assertEqual(sorted(set(lines)), sorted(lines))