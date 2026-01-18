from openstack.message.v2 import claim as _claim
from openstack.message.v2 import message as _message
from openstack.message.v2 import queue as _queue
from openstack.message.v2 import subscription as _subscription
from openstack import proxy
from openstack import resource
def update_claim(self, queue_name, claim, **attrs):
    """Update an existing claim from attributes

        :param queue_name: The name of target queue to claim message from.
        :param claim: The value can be either the ID of a claim or a
            :class:`~openstack.message.v2.claim.Claim` instance.
        :param dict attrs: Keyword arguments which will be used to update a
            :class:`~openstack.message.v2.claim.Claim`,
            comprised of the properties on the Claim class.

        :returns: The results of claim update
        :rtype: :class:`~openstack.message.v2.claim.Claim`
        """
    return self._update(_claim.Claim, claim, queue_name=queue_name, **attrs)