from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def ShowStructuredOutput(self):
    """Returns True if output should be Structured, False otherwise."""
    show_messages = properties.VALUES.core.show_structured_logs.Get()
    if any([show_messages == 'terminal' and self.terminal, show_messages == 'log' and (not self.terminal), show_messages == 'always']):
        return True
    return False