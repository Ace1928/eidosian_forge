from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetFromFallthroughs(fallthroughs, parsed_args, attribute_name=None):
    """Gets the value of an attribute based on fallthrough information.

    If the attribute value is not provided by any of the fallthroughs, an
    error is raised with a list of ways to provide information about the
    attribute.

  Args:
    fallthroughs: [_FallthroughBase], list of fallthroughs.
    parsed_args: a parsed argparse namespace.
    attribute_name: str, the name of the attribute. Used for error message,
      omitted if not provided.

  Returns:
    the value of the attribute.

  Raises:
    AttributeNotFoundError: if no value can be found.
  """
    for fallthrough in fallthroughs:
        try:
            return fallthrough.GetValue(parsed_args)
        except FallthroughNotFoundError:
            continue
    hints = GetHints(fallthroughs)
    fallthroughs_summary = '\n'.join(['- {}'.format(hint) for hint in hints])
    raise AttributeNotFoundError('Failed to find attribute{}. The attribute can be set in the following ways: \n{}'.format('' if attribute_name is None else ' [{}]'.format(attribute_name), fallthroughs_summary))