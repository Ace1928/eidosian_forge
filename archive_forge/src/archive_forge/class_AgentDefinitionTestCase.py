from neutron_lib.api.definitions import agent
from neutron_lib.api.definitions import agent_sort_key
from neutron_lib.tests.unit.api.definitions import base
class AgentDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = agent_sort_key
    extension_resources = (agent.COLLECTION_NAME,)
    extension_attributes = ('topic', 'agent_type', 'created_at', 'configurations', 'heartbeat_timestamp', 'binary', 'started_at', 'host', 'alive')