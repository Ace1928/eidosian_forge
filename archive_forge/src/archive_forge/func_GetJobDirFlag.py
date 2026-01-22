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
def GetJobDirFlag(upload_help=True, allow_local=False):
    """Get base.Argument() for `--job-dir`.

  If allow_local is provided, this Argument gives a str when parsed; otherwise,
  it gives a (possibly empty) ObjectReference.

  Args:
    upload_help: bool, whether to include help text related to object upload.
      Only useful in remote situations (`jobs submit training`).
    allow_local: bool, whether to allow local directories (only useful in local
      situations, like `local train`) or restrict input to directories in Cloud
      Storage.

  Returns:
    base.Argument() for the corresponding `--job-dir` flag.
  """
    help_ = '{dir_type} in which to store training outputs and other data\nneeded for training.\n\nThis path will be passed to your TensorFlow program as the `--job-dir` command-line\narg. The benefit of specifying this field is that AI Platform will validate\nthe path for use in training. However, note that your training program will need\nto parse the provided `--job-dir` argument.\n'.format(dir_type='Cloud Storage path' + (' or local_directory' if allow_local else ''))
    if upload_help:
        help_ += '\nIf packages must be uploaded and `--staging-bucket` is not provided, this path\nwill be used instead.\n'
    if allow_local:
        type_ = str
    else:
        type_ = functools.partial(storage_util.ObjectReference.FromArgument, allow_empty_object=True)
    return base.Argument('--job-dir', type=type_, help=help_)