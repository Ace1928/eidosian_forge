import abc
from neutron_lib.api.definitions import portbindings
def update_subnet_precommit(self, context):
    """Update resources of a subnet.

        :param context: SubnetContext instance describing the new
            state of the subnet, as well as the original state prior
            to the update_subnet call.

        Update values of a subnet, updating the associated resources
        in the database. Called inside transaction context on session.
        Raising an exception will result in rollback of the
        transaction.

        update_subnet_precommit is called for all changes to the
        subnet state. It is up to the mechanism driver to ignore
        state or state changes that it does not know or care about.
        """
    pass