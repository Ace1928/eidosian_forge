import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def MapSimulationTypeToRequest(api_version, config_changes_list, simulation_type):
    """Parse and map the appropriate simulation type to request."""
    if not config_changes_list:
        print('No new changes to simulate.')
        exit()
    if simulation_type == 'shadowed-firewall':
        return Messages(api_version).Simulation(configChanges=config_changes_list, shadowedFirewallSimulationData=Messages(api_version).ShadowedFirewallSimulationData())
    if simulation_type == 'connectivity-test':
        return Messages(api_version).Simulation(configChanges=config_changes_list, connectivityTestSimulationData=Messages(api_version).ConnectivityTestSimulationData())
    raise InvalidInputError('Invalid simulation-type.')