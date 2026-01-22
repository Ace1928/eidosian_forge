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
def ParseMsg(self, msg):
    """Converts msg to a console safe pair of plain and ANSI-annotated strings.

    Args:
      msg: str or text.TypedText, the message to parse into plain and
        ANSI-annotated strings.
    Returns:
      str, str: A plain text string and a string that may also contain ANSI
        constrol sequences. If ANSI is not supported or color is disabled,
        then the second string will be identical to the first.
    """
    plain_text, styled_text = (msg, msg)
    if isinstance(msg, text.TypedText):
        typed_text_parser = style_parser.GetTypedTextParser()
        plain_text = typed_text_parser.ParseTypedTextToString(msg, stylize=False)
        styled_text = typed_text_parser.ParseTypedTextToString(msg, stylize=self.isatty())
    plain_text = console_attr.SafeText(plain_text, encoding=LOG_FILE_ENCODING, escape=False)
    styled_text = console_attr.SafeText(styled_text, encoding=LOG_FILE_ENCODING, escape=False)
    return (plain_text, styled_text)