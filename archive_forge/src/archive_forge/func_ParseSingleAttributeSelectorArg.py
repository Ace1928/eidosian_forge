from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import Collection
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
def ParseSingleAttributeSelectorArg(arg_name, arg_value: Collection[str]):
    """Parses a single attribute selector argument."""
    _, messages = util.GetClientAndMessages()
    single_attribute_selector_matcher = re.compile('([^=]+)(?:=)(.+)', re.DOTALL)
    single_attribute_selectors = []
    for arg in arg_value:
        match = single_attribute_selector_matcher.match(arg)
        if not match:
            raise gcloud_exceptions.InvalidArgumentException(arg_name, 'Invalid flag value [{0}]'.format(arg))
        single_attribute_selectors.append(messages.SingleAttributeSelector(attribute=match.group(1), value=match.group(2)))
    return single_attribute_selectors