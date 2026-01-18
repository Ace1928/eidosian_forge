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
def test_prompt_for_confirmation(self):
    """Test :func:`humanfriendly.prompts.prompt_for_confirmation()`."""
    for reply in ('yes', 'Yes', 'YES', 'y', 'Y'):
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: reply):
            assert prompt_for_confirmation('Are you sure?') is True
    for reply in ('no', 'No', 'NO', 'n', 'N'):
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: reply):
            assert prompt_for_confirmation('Are you sure?') is False
    for default_choice in (True, False):
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
            assert prompt_for_confirmation('Are you sure?', default=default_choice) is default_choice
    replies = ['', 'y']
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
        with CaptureOutput(merged=True) as capturer:
            assert prompt_for_confirmation('Are you sure?') is True
            assert "there's no default choice" in capturer.get_text()
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'y'):
        for default_value, expected_text in ((True, 'Y/n'), (False, 'y/N'), (None, 'y/n')):
            with CaptureOutput(merged=True) as capturer:
                assert prompt_for_confirmation('Are you sure?', default=default_value) is True
                assert expected_text in capturer.get_text()
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
        with self.assertRaises(TooManyInvalidReplies):
            prompt_for_confirmation('Are you sure?')