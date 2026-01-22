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
class _ManCommandCollector(_ManPageCollector):
    """man command help document section collector."""
    _CLI_VERSION = 'man-0.1'

    @classmethod
    def GetVersion(cls):
        return cls._CLI_VERSION

    def _GetRawManPageText(self):
        """Returns the raw man page text."""
        try:
            with files.FileWriter(os.devnull) as f:
                return encoding.Decode(subprocess.check_output(['man', self.command_name], stderr=f))
        except (OSError, subprocess.CalledProcessError):
            raise NoManPageTextForCommandError('Cannot get man(1) command man page text for [{}].'.format(self.command_name))

    def GetManPageText(self):
        """Returns the preprocessed man page text."""
        text = self._GetRawManPageText()
        return re.sub('.\x08', '', re.sub('(‐|\\u2010)\n *', '', text))