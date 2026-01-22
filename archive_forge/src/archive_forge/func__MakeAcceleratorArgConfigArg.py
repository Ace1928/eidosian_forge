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
def _MakeAcceleratorArgConfigArg(arg_name, arg_help, required=False):
    """Creates an ArgDict representing an AcceleratorConfig message."""
    type_help = '*type*::: Type of the accelerator. Choices are {}'.format(','.join(_ACCELERATOR_TYPE_MAPPER.choices))
    count_help = '*count*::: Number of accelerators to attach to each machine running the job. Must be greater than 0.'
    arg = base.Argument(arg_name, required=required, type=arg_parsers.ArgDict(spec={'type': _ConvertAcceleratorTypeToEnumValue, 'count': _ValidateAcceleratorCount}, required_keys=['type', 'count']), help='{arg_help}\n\n{type_help}\n\n{count_help}\n\n'.format(arg_help=arg_help, type_help=type_help, count_help=count_help))
    return arg