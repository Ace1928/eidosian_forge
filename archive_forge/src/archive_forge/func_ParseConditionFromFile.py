from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def ParseConditionFromFile(condition_from_file):
    """Read condition from YAML or JSON file."""
    condition = arg_parsers.FileContents()(condition_from_file)
    condition_dict = iam_util.ParseYamlOrJsonCondition(condition, _ConditionFileFormatException(condition_from_file))
    return condition_dict