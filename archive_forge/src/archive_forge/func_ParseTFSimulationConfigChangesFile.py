import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def ParseTFSimulationConfigChangesFile(proposed_changes_file, api_version, simulation_type):
    """Parses and converts the config changes file into API Format."""
    try:
        tf_plan = yaml.load_path(proposed_changes_file)
    except yaml.YAMLParseError as unused_ref:
        raise InvalidFileError('Error parsing config changes file: [{}]'.format(proposed_changes_file))
    enum = Messages(api_version).ConfigChange.UpdateTypeValueValuesEnum
    update_resources_list = []
    delete_resources_list = []
    tainted_resources_list = []
    supported_resource_types = ['google_compute_firewall']
    for resource_change_config in tf_plan['resource_changes']:
        if resource_change_config['type'] not in supported_resource_types:
            continue
        resource_change_object = resource_change_config['change']
        actions = resource_change_object['actions']
        if resource_change_object['before']:
            resource_self_link = resource_change_object['before']['self_link']
        if len(actions) > 1:
            tainted_resources_list.append(resource_self_link)
        elif 'update' in actions:
            update_resources_list.append(resource_self_link)
        elif 'delete' in actions:
            delete_resources_list.append(resource_self_link)
    gcp_config = ParseTerraformPlanFileTFTools(proposed_changes_file)
    config_changes_list = ParseAndAddResourceConfigToConfigChangesList(gcp_config, tainted_resources_list, update_resources_list, enum, api_version)
    config_changes_list = AddDeleteResourcesToConfigChangesList(delete_resources_list, config_changes_list, enum, api_version)
    return MapSimulationTypeToRequest(api_version, config_changes_list, simulation_type)