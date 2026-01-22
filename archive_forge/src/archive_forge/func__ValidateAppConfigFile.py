from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import run_apps_operations
from googlecloudsdk.command_lib.runapps import exceptions
def _ValidateAppConfigFile(self, file_content):
    if 'name' not in file_content and 'resources' not in file_content:
        raise exceptions.FieldMismatchError("'name' or 'resources' is missing.")
    if '/t' in file_content:
        raise exceptions.ConfigurationError('tabs found in manifest content.')