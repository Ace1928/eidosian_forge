from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import run_apps_operations
def _SetFormat(namespace: parser_extensions.Namespace) -> None:
    columns = ['formatted_latest_deployment_status:label=', 'integration_name:label=INTEGRATION', 'integration_type:label=TYPE', 'region:label=REGION', 'services:label=SERVICE']
    namespace.GetDisplayInfo().AddFormat('table({columns})'.format(columns=','.join(columns)))