from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
import random
import re
import socket
import subprocess
import tempfile
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
def BuildArgsList(args):
    """Converts an argparse.Namespace to a list of arg strings."""
    args_list = []
    if args.host_port:
        if _IPV6_RE.match(args.host_port.host):
            host = '[{}]'.format(args.host_port.host)
        else:
            host = args.host_port.host
        if args.host_port.host is not None:
            args_list.append('--host={0}'.format(host))
        if args.host_port.port is not None:
            args_list.append('--port={0}'.format(args.host_port.port))
    return args_list