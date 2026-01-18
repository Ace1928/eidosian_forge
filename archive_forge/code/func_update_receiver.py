from openstack.clustering.v1 import action as _action
from openstack.clustering.v1 import build_info
from openstack.clustering.v1 import cluster as _cluster
from openstack.clustering.v1 import cluster_attr as _cluster_attr
from openstack.clustering.v1 import cluster_policy as _cluster_policy
from openstack.clustering.v1 import event as _event
from openstack.clustering.v1 import node as _node
from openstack.clustering.v1 import policy as _policy
from openstack.clustering.v1 import policy_type as _policy_type
from openstack.clustering.v1 import profile as _profile
from openstack.clustering.v1 import profile_type as _profile_type
from openstack.clustering.v1 import receiver as _receiver
from openstack.clustering.v1 import service as _service
from openstack import proxy
from openstack import resource
def update_receiver(self, receiver, **attrs):
    """Update a receiver.

        :param receiver: The value can be either the name or ID of a receiver
            or a :class:`~openstack.clustering.v1.receiver.Receiver` instance.
        :param attrs: The attributes to update on the receiver parameter.
            Valid attribute names include ``name``, ``action`` and ``params``.
        :returns: The updated receiver.
        :rtype: :class:`~openstack.clustering.v1.receiver.Receiver`
        """
    return self._update(_receiver.Receiver, receiver, **attrs)