from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import enum
import getpass
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_pager
from googlecloudsdk.core.console import prompt_completer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def PromptPassword(prompt, error_message='Invalid Password', validation_callable=None, encoder_callable=None):
    """Prompt user for password with optional validation."""
    if properties.VALUES.core.disable_prompts.GetBool():
        return None
    str_prompt = six.ensure_str(prompt)
    pass_wd = getpass.getpass(str_prompt)
    encoder = encoder_callable if callable(encoder_callable) else six.ensure_str
    if callable(validation_callable):
        while not validation_callable(pass_wd):
            sys.stderr.write(_DoWrap(error_message) + '\n')
            pass_wd = getpass.getpass(str_prompt)
    return encoder(pass_wd)