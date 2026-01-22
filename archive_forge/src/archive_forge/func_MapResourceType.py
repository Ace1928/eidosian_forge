import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def MapResourceType(resource_type):
    if resource_type == 'compute#firewall':
        return 'compute.googleapis.com/Firewall'
    raise InvalidInputError('Only Firewall resources are supported. Instead found {}'.format(resource_type))