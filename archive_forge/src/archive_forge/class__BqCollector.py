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
class _BqCollector(object):
    """bq help document section collector."""

    def __init__(self, text):
        self.text = text.split('\n')
        self.heading = 'DESCRIPTION'
        self.lookahead = None
        self.ignore_trailer = False

    def Collect(self, strip_headings=False):
        """Returns the heading and content lines from text."""
        content = []
        if self.lookahead:
            if not strip_headings:
                content.append(self.lookahead)
            self.lookahead = None
        heading = self.heading
        self.heading = None
        while self.text:
            line = self.text.pop(0)
            if line.startswith(' ') or (not strip_headings and (not self.ignore_trailer)):
                content.append(line.rstrip())
        while content and (not content[0]):
            content.pop(0)
        while content and (not content[-1]):
            content.pop()
        self.ignore_trailer = True
        return (heading, content)