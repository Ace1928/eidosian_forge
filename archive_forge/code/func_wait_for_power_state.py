import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def wait_for_power_state(self, session, expected_state, timeout=None):
    """Wait for the node to reach the expected power state.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param expected_state: The expected power state to reach.
        :param timeout: If ``wait`` is set to ``True``, specifies how much (in
            seconds) to wait for the expected state to be reached. The value of
            ``None`` (the default) means no client-side timeout.

        :return: This :class:`Node` instance.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        """
    for count in utils.iterate_timeout(timeout, "Timeout waiting for node %(node)s to reach power state '%(state)s'" % {'node': self.id, 'state': expected_state}):
        self.fetch(session)
        if self.power_state == expected_state:
            return self
        session.log.debug('Still waiting for node %(node)s to reach power state "%(target)s", the current state is "%(state)s"', {'node': self.id, 'target': expected_state, 'state': self.power_state})