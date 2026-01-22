from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import getpass
import io
import locale
import os
import platform as system_platform
import re
import ssl
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import http_proxy_setup
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import requests
import six
import urllib3
class ToolsInfo(object):
    """Holds info about tools gcloud interacts with."""

    def __init__(self, anonymize=None):
        del anonymize
        self.git_version = self._GitVersion()
        self.ssh_version = self._SshVersion()

    def _GitVersion(self):
        return self._GetVersion(['git', '--version'])

    def _SshVersion(self):
        return self._GetVersion(['ssh', '-V'])

    def _GetVersion(self, cmd):
        """Return tools version."""
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError:
            return 'NOT AVAILABLE'
        stdoutdata, _ = proc.communicate()
        data = [f for f in stdoutdata.split(b'\n') if f]
        if len(data) != 1:
            return 'NOT AVAILABLE'
        else:
            return encoding.Decode(data[0])

    def __str__(self):
        return textwrap.dedent('        git: [{git}]\n        ssh: [{ssh}]\n        '.format(git=self.git_version, ssh=self.ssh_version))