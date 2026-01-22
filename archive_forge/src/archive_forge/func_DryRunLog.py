from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import subprocess
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
import six
def DryRunLog(msg):
    log.out.Print('[COMMAND TO RUN]: %s\n' % msg)