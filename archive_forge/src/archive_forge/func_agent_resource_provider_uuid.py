import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def agent_resource_provider_uuid(namespace, host):
    """Generate a stable UUID for an agent.

    :param namespace: A UUID object identifying a mechanism driver (including
                      its agent).
    :param host: The hostname of the agent.
    :returns: A unique and stable UUID identifying an agent.
    """
    return six_uuid5(namespace=namespace, name=host)