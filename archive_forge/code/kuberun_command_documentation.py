from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import auth
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberuncli
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
Processes the result of a successful kuberun command execution.

    Child classes typically override this method to parse and return a
    structured result (e.g. JSON). The returned data object will be passed
    through cloudsdk filtering/sorting (if applicable) and rendered in the
    default or user-specified output format.

    Args:
      out: str, the output of the kuberun command
      args: the arguments passed to the gcloud command

    Returns:
      A resource object dispatched by display.Displayer().
    