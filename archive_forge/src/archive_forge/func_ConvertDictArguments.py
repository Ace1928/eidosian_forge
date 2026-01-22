from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ConvertDictArguments(self, arguments, value_message):
    """Convert dictionary arguments to parameter list .

    Args:
      arguments: Arguments for create job using template.
      value_message: the value message of the arguments

    Returns:
      List of value_message.AdditionalProperty
    """
    params_list = []
    if arguments:
        for k, v in six.iteritems(arguments):
            params_list.append(value_message.AdditionalProperty(key=k, value=v))
    return params_list