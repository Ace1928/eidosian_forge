from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import resources
import six
def LowerCaseType(value):
    return value.lower()