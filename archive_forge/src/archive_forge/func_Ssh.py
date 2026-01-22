from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import socket
import ssl
import sys
import threading
import time
from apitools.base.py.exceptions import Error
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.api_lib.workstations.util import GetClientInstance
from googlecloudsdk.api_lib.workstations.util import GetMessagesModule
from googlecloudsdk.api_lib.workstations.util import VERSION_MAP
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from requests import certs
import six
import websocket
import websocket._exceptions as websocket_exceptions
def Ssh(self, args):
    """SSH's to a workstation."""
    self.env = ssh.Environment.Current()
    self.env.RequireSSH()
    keys = ssh.Keys.FromFilename()
    keys.EnsureKeysExist(overwrite=False)
    host, port = self._GetLocalHostPort(args)
    remote = ssh.Remote(host=host, user=args.user)
    port = args.local_host_port.port if int(args.local_host_port.port) != 0 else six.text_type(self.socket.getsockname()[1])
    options = {'UserKnownHostsFile': '/dev/null', 'StrictHostKeyChecking': 'no', 'ServerAliveInterval': '0'}
    remainder = []
    if args.ssh_args:
        remainder.extend(args.ssh_args)
    tty = not args.command
    command_list = args.command.split(' ') if args.command else None
    remote_command = containers.GetRemoteCommand(None, command_list)
    cmd = ssh.SSHCommand(remote=remote, port=port, options=options, tty=tty, remainder=remainder, remote_command=remote_command)
    return cmd.Run(self.env)