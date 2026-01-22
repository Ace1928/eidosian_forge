from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.core import exceptions
class NoFieldsSpecifiedError(exceptions.Error):
    """Raises when no arguments specified for update commands."""