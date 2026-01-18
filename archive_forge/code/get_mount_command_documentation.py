from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.cloud_shell import util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
Prints a command to mount the Cloud Shell home directory via sshfs.