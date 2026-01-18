import abc
from neutron_lib.api.definitions import portbindings
def update_network_postcommit(self, context):
    """Update a network.

        :param context: NetworkContext instance describing the new
            state of the network, as well as the original state prior
            to the update_network call.

        Called after the transaction commits. Call can block, though
        will block the entire process so care should be taken to not
        drastically affect performance. Raising an exception will
        cause the deletion of the resource.

        update_network_postcommit is called for all changes to the
        network state.  It is up to the mechanism driver to ignore
        state or state changes that it does not know or care about.
        """
    pass