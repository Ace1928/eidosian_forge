from openstack.cloud import _utils
from openstack import exceptions
def update_coe_cluster(self, name_or_id, **kwargs):
    """Update a COE cluster.

        :param name_or_id: Name or ID of the COE cluster being updated.
        :param kwargs: Cluster attributes to be updated.

        :returns: The updated cluster ``Cluster`` object.

        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    cluster = self.get_coe_cluster(name_or_id)
    if not cluster:
        raise exceptions.SDKException('COE cluster %s not found.' % name_or_id)
    cluster = self.container_infrastructure_management.update_cluster(cluster, **kwargs)
    return cluster