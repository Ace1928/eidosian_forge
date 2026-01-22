from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib import genomics as lib
from googlecloudsdk.api_lib.genomics import exceptions
from googlecloudsdk.api_lib.genomics import genomics_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _ValidateAndMergeArgInputs(args):
    """Turn args.inputs and args.inputs_from_file dicts into a single dict.

  Args:
    args: The parsed command-line arguments

  Returns:
    A dict that is the merge of args.inputs and args.inputs_from_file
  Raises:
    files.Error
  """
    is_local_file = {}
    if not args.inputs_from_file:
        return (args.inputs, is_local_file)
    arg_inputs = {}
    if args.inputs:
        overlap = set(args.inputs.keys()).intersection(set(args.inputs_from_file.keys()))
        if overlap:
            raise exceptions.GenomicsError('--{0} and --{1} may not specify overlapping values: {2}'.format('inputs', 'inputs-from-file', ', '.join(overlap)))
        arg_inputs.update(args.inputs)
    for key, value in six.iteritems(args.inputs_from_file):
        arg_inputs[key] = files.ReadFileContents(value)
        is_local_file[key] = True
    return (arg_inputs, is_local_file)