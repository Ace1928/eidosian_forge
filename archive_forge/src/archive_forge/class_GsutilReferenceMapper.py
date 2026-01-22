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
class GsutilReferenceMapper(CommandReferenceMapper):
    """gsutil help reference mapper."""

    def __init__(self, cli, args):
        super(GsutilReferenceMapper, self).__init__(cli, args)
        self.subcommand = args[1] if len(args) > 1 else ''
        self.ref = ['https://cloud.google.com/storage/docs/gsutil']

    def GetMan(self):
        cmd = ['gsutil help']
        if self.subcommand:
            cmd.append(self.subcommand)
        cmd.append('| less')
        return ' '.join(cmd)

    def GetURL(self):
        if self.subcommand:
            self.ref.append('commands')
            self.ref.append(self.subcommand)
        return '/'.join(self.ref)