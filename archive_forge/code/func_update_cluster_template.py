from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def update_cluster_template(self, cluster_template, **attrs):
    """Update a cluster_template

        :param cluster_template: Either the id of a cluster_template or a
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
            instance.
        :param attrs: The attributes to update on the cluster_template
            represented by ``cluster_template``.

        :returns: The updated cluster_template
        :rtype:
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
        """
    return self._update(_cluster_template.ClusterTemplate, cluster_template, **attrs)