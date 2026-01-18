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
def test_show_pager(self):
    """Test :func:`humanfriendly.terminal.show_pager()`."""
    original_pager = os.environ.get('PAGER', None)
    try:
        os.environ['PAGER'] = 'cat'
        random_text = '\n'.join((random_string(25) for i in range(50)))
        with CaptureOutput() as capturer:
            show_pager(random_text)
            assert random_text in capturer.get_text()
    finally:
        if original_pager is not None:
            os.environ['PAGER'] = original_pager
        else:
            os.environ.pop('PAGER')