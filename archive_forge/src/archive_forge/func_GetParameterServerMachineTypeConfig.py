from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import functools
import itertools
import sys
import textwrap
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.ml_engine import constants
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetParameterServerMachineTypeConfig():
    """Build parameter-server machine type config group."""
    machine_type = base.Argument('--parameter-server-machine-type', required=True, help="Type of virtual machine to use for training job's parameter servers. This flag must be specified if any of the other arguments in this group are specified machine to use for training job's parameter servers.")
    machine_count = base.Argument('--parameter-server-count', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), required=True, help='Number of parameter servers to use for the training job.')
    machine_type_group = base.ArgumentGroup(required=False, help='Configure parameter server machine type settings.')
    machine_type_group.AddArgument(machine_type)
    machine_type_group.AddArgument(machine_count)
    return machine_type_group