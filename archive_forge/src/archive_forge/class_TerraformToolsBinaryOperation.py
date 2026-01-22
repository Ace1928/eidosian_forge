import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
class TerraformToolsBinaryOperation(binary_operations.BinaryBackedOperation):
    """BinaryOperation for Terraform Tools binary."""
    custom_errors = {}

    def __init__(self, **kwargs):
        super(TerraformToolsBinaryOperation, self).__init__(binary='terraform-tools', check_hidden=True, install_if_missing=True, custom_errors=None, **kwargs)

    def _ParseArgsForCommand(self, command, terraform_plan_json, project, verbosity='debug', **kwargs):
        args = [command, terraform_plan_json, '--verbosity', verbosity]
        if project:
            args += ['--project', project]
        return args