import re
import types
from typing import FrozenSet, Optional, Tuple
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.functions.v2 import types as api_types
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.eventarc import types as trigger_types
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import labels_util
from googlecloudsdk.command_lib.functions import run_util
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.command_lib.functions.v2 import deploy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
def _GetVpcAndVpcEgressSettings(args: parser_extensions.Namespace, messages: types.ModuleType, existing_function) -> Tuple[Optional[str], Optional[api_types.VpcConnectorEgressSettings], FrozenSet[str]]:
    """Constructs vpc connector and egress settings from command-line arguments.

  Args:
    args: The arguments that this command was invoked with.
    messages: messages module, the GCFv2 message stubs.
    existing_function: The pre-existing function.

  Returns:
    A tuple `(vpc_connector, egress_settings, updated_fields_set)` where:
    - `vpc_connector` is the name of the vpc connector,
    - `egress_settings` is the egress settings for the vpc connector,
    - `updated_fields_set` is the set of update mask fields.
  """
    if args.clear_vpc_connector:
        return (None, None, frozenset(['service_config.vpc_connector', 'service_config.vpc_connector_egress_settings']))
    update_fields_set = set()
    vpc_connector = None
    if args.vpc_connector:
        vpc_connector = args.CONCEPTS.vpc_connector.Parse().RelativeName()
        update_fields_set.add('service_config.vpc_connector')
    elif existing_function and existing_function.serviceConfig and existing_function.serviceConfig.vpcConnector:
        vpc_connector = existing_function.serviceConfig.vpcConnector
    egress_settings = None
    if args.egress_settings:
        if not vpc_connector:
            raise exceptions.RequiredArgumentException('vpc-connector', 'Flag `--vpc-connector` is required for setting `--egress-settings`.')
        egress_settings = arg_utils.ChoiceEnumMapper(arg_name='egress_settings', message_enum=messages.ServiceConfig.VpcConnectorEgressSettingsValueValuesEnum, custom_mappings=flags.EGRESS_SETTINGS_MAPPING).GetEnumForChoice(args.egress_settings)
        update_fields_set.add('service_config.vpc_connector_egress_settings')
    return (vpc_connector, egress_settings, frozenset(update_fields_set))