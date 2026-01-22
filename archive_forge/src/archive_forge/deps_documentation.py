from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
Initializes a fallthrough getting a parameter from the anchor.

    For anchor arguments which can be plural, returns the list.

    Args:
      fallthroughs: list[_FallthroughBase], any fallthrough for an anchor arg.
      collection_info: the info of the collection to parse the anchor as.
      parameter_name: str, the name of the parameter
      plural: bool, whether the expected result should be a list. Should be
        False for everything except the "anchor" arguments in a case where a
    