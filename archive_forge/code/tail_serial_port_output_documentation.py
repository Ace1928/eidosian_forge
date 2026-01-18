from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
Periodically fetch new output from a virtual machine instance's serial port and display it as it becomes available.

  {command} is used to tail the output from a Compute
  Engine virtual machine instance's serial port. The serial port output
  from the instance will be printed to standard output. This
  information can be useful for diagnostic purposes.
  