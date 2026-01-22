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
def AddPythonVersionFlag(parser, context):
    help_str = '      Version of Python used {context}. Choices are 3.7, 3.5, and 2.7.\n      However, this value must be compatible with the chosen runtime version\n      for the job.\n\n      Must be used with a compatible runtime version:\n\n      * 3.7 is compatible with runtime versions 1.15 and later.\n      * 3.5 is compatible with runtime versions 1.4 through 1.14.\n      * 2.7 is compatible with runtime versions 1.15 and earlier.\n      '.format(context=context)
    version = base.Argument('--python-version', help=help_str)
    version.AddToParser(parser)