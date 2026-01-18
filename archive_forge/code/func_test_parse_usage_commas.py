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
def test_parse_usage_commas(self):
    """Test :func:`humanfriendly.usage.parse_usage()` against option labels containing commas."""
    introduction, options = self.preprocess_parse_result("\n            Usage: my-fancy-app [OPTIONS]\n\n            Some introduction goes here.\n\n            Supported options:\n\n              -f, --first-option\n\n                Explanation of first option.\n\n              -s, --second-option=WITH,COMMA\n\n                This should be a separate option's description.\n        ")
    assert 'Usage: my-fancy-app [OPTIONS]' in introduction
    assert 'Some introduction goes here.' in introduction
    assert 'Supported options:' in introduction
    assert '-f, --first-option' in options
    assert 'Explanation of first option.' in options
    assert '-s, --second-option=WITH,COMMA' in options
    assert "This should be a separate option's description." in options