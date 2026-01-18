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
def test_smart_tables(self):
    """Test :func:`humanfriendly.tables.format_smart_table()`."""
    column_names = ['One', 'Two', 'Three']
    data = [['1', '2', '3'], ['a', 'b', 'c']]
    assert ansi_strip(format_smart_table(data, column_names)) == dedent('\n            ---------------------\n            | One | Two | Three |\n            ---------------------\n            | 1   | 2   | 3     |\n            | a   | b   | c     |\n            ---------------------\n        ').strip()
    column_names = ['One', 'Two', 'Three']
    data = [['1', '2', '3'], ['a', 'b', 'Here comes a\nmulti line column!']]
    assert ansi_strip(format_smart_table(data, column_names)) == dedent('\n            ------------------\n            One: 1\n            Two: 2\n            Three: 3\n            ------------------\n            One: a\n            Two: b\n            Three:\n            Here comes a\n            multi line column!\n            ------------------\n        ').strip()