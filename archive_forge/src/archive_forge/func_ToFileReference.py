from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.cloud_shell import util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
import six
def ToFileReference(path, remote):
    if path.startswith('cloudshell:'):
        return ssh.FileReference.FromPath(path.replace('cloudshell', six.text_type(remote), 1))
    elif path.startswith('localhost:'):
        return ssh.FileReference.FromPath(path.replace('localhost:', '', 1))
    else:
        raise Exception('invalid path: ' + path)