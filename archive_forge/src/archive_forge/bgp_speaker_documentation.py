from openstack import exceptions
from openstack import resource
from openstack import utils
Delete BGP Speaker from a Dynamic Routing Agent

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param bgp_agent_id: The id of the dynamic routing agent from which
                             remove the speaker.
        