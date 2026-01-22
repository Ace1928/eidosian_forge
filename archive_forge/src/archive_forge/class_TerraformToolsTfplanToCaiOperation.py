from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
class TerraformToolsTfplanToCaiOperation(binary_operations.StreamingBinaryBackedOperation):
    """Streaming operation for Terraform Tools tfplan-to-cai command."""
    custom_errors = {}

    def __init__(self, **kwargs):
        custom_errors = {'MISSING_EXEC': MISSING_BINARY.format(binary='terraform-tools')}
        super(TerraformToolsTfplanToCaiOperation, self).__init__(binary='terraform-tools', check_hidden=True, install_if_missing=True, custom_errors=custom_errors, structured_output=True, **kwargs)

    def _ParseArgsForCommand(self, command, terraform_plan_json, project, region, zone, verbosity, output_path, **kwargs):
        args = [command, terraform_plan_json, '--output-path', output_path, '--verbosity', verbosity, '--user-agent', metrics.GetUserAgent()]
        if project:
            args += ['--project', project]
        if region:
            args += ['--region', region]
        if zone:
            args += ['--zone', zone]
        return args