from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import subprocess
import sys
import webbrowser
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
class CommandReferenceMapper(six.with_metaclass(abc.ABCMeta, object)):
    """Command to URL or man page reference mapper base class."""

    def __init__(self, cli, args):
        self.cli = cli
        self.args = args

    @abc.abstractmethod
    def GetMan(self):
        """Returns the man-style command for the command in args."""
        return None

    @abc.abstractmethod
    def GetURL(self):
        """Returns the help doc URL for the command in args."""
        return None