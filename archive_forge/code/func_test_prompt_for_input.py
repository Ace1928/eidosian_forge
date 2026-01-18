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
def test_prompt_for_input(self):
    """Test :func:`humanfriendly.prompts.prompt_for_input()`."""
    with open(os.devnull) as handle:
        with PatchedAttribute(sys, 'stdin', handle):
            default_value = 'To seek the holy grail!'
            assert prompt_for_input('What is your quest?', default=default_value) == default_value
            with self.assertRaises(EOFError):
                prompt_for_input('What is your favorite color?')