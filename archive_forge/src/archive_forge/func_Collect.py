from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
def Collect(self):
    """Returns the heading and content lines from text."""
    content = []
    heading = self.heading
    self.heading = None
    while self.text:
        line = self.text.pop(0)
        if not heading:
            if line == 'NAME':
                heading = line
            continue
        elif not line:
            pass
        elif line[0] == ' ':
            if not self.content_indent:
                self.content_indent = re.sub('[^ ].*', '', line)
            if len(line) > len(self.content_indent):
                indented_char = line[len(self.content_indent)]
                if not line.startswith(self.content_indent):
                    line = '### ' + line.strip()
                elif heading == 'DESCRIPTION' and indented_char == '-':
                    self.text.insert(0, line)
                    self.heading = 'FLAGS'
                    break
                elif heading == 'FLAGS' and indented_char not in (' ', '-'):
                    self.text.insert(0, line)
                    self.heading = 'DESCRIPTION'
                    break
        elif line in ('SYNOPSIS', 'DESCRIPTION', 'EXIT STATUS', 'SEE ALSO'):
            self.heading = line
            break
        elif 'FLAGS' in line or 'OPTIONS' in line:
            self.heading = 'FLAGS'
            break
        elif line and line[0].isupper():
            self.heading = line.split(' ', 1)[-1]
            break
        content.append(line.rstrip())
    while content and (not content[0]):
        content.pop(0)
    while content and (not content[-1]):
        content.pop()
    return (heading, content)