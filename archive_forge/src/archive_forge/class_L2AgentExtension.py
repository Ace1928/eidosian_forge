import abc
from neutron_lib.agent import extension
class L2AgentExtension(extension.AgentExtension, metaclass=abc.ABCMeta):
    """Define stable abstract interface for l2 agent extensions.

    An agent extension extends the agent core functionality.
    """

    def initialize(self, connection, driver_type):
        """Initialize agent extension.

        :param connection: RPC connection that can be reused by the extension
                           to define its RPC endpoints
        :param driver_type: String that defines the agent type to the
                            extension. Can be used to choose the right backend
                            implementation.
        """

    @abc.abstractmethod
    def handle_port(self, context, data):
        """Handle a port add/update event.

        This can be called on either create or update, depending on the
        code flow. Thus, it's this function's responsibility to check what
        actually changed.

        :param context: RPC context.
        :param data: Port data.
        """

    @abc.abstractmethod
    def delete_port(self, context, data):
        """Handle a port delete event.

        :param context: RPC context.
        :param data: Port data.
        """