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
def test_pretty_tables(self):
    """Test :func:`humanfriendly.tables.format_pretty_table()`."""
    data = [['Just one column']]
    assert format_pretty_table(data) == dedent('\n            -------------------\n            | Just one column |\n            -------------------\n        ').strip()
    data = [['One', 'Two', 'Three'], ['1', '2', '3']]
    assert format_pretty_table(data) == dedent('\n            ---------------------\n            | One | Two | Three |\n            | 1   | 2   | 3     |\n            ---------------------\n        ').strip()
    column_names = ['One', 'Two', 'Three']
    data = [['1', '2', '3'], ['a', 'b', 'c']]
    assert ansi_strip(format_pretty_table(data, column_names)) == dedent('\n            ---------------------\n            | One | Two | Three |\n            ---------------------\n            | 1   | 2   | 3     |\n            | a   | b   | c     |\n            ---------------------\n        ').strip()
    column_names = ['Just a label', 'Important numbers']
    data = [['Row one', '15'], ['Row two', '300']]
    assert ansi_strip(format_pretty_table(data, column_names)) == dedent('\n            ------------------------------------\n            | Just a label | Important numbers |\n            ------------------------------------\n            | Row one      |                15 |\n            | Row two      |               300 |\n            ------------------------------------\n        ').strip()