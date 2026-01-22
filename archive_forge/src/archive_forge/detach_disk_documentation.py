from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import resource_args
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
Set up arguments for this command.

    Args:
      parser: An argparse.ArgumentParser.
    