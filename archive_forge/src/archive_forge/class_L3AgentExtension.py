import abc
from neutron_lib.agent import extension
class L3AgentExtension(extension.AgentExtension, metaclass=abc.ABCMeta):
    """Define stable abstract interface for l3 agent extensions.

    An agent extension extends the agent core functionality.
    """

    @abc.abstractmethod
    def add_router(self, context, data):
        """Handle a router add event.

        Called on router create.

        :param context: RPC context.
        :param data: Router data.
        """

    @abc.abstractmethod
    def update_router(self, context, data):
        """Handle a router update event.

        Called on router update.

        :param context: RPC context.
        :param data: Router data.
        """

    @abc.abstractmethod
    def delete_router(self, context, data):
        """Handle a router delete event.

        :param context: RPC context.
        :param data: Router data.
        """

    @abc.abstractmethod
    def ha_state_change(self, context, data):
        """Change router state from agent extension.

        Called on HA router state change.

        :param context: rpc context
        :param data: dict of router_id and new state
        """

    @abc.abstractmethod
    def update_network(self, context, data):
        """Handle a network update event.

        Called on network update.

        :param context: RPC context.
        :param data: dict of network data.
        """