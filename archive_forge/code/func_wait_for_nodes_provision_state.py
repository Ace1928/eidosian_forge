from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def wait_for_nodes_provision_state(self, nodes, expected_state, timeout=None, abort_on_failed_state=True, fail=True):
    """Wait for the nodes to reach the expected state.

        :param nodes: List of nodes - name, ID or
            :class:`~openstack.baremetal.v1.node.Node` instance.
        :param expected_state: The expected provisioning state to reach.
        :param timeout: If ``wait`` is set to ``True``, specifies how much (in
            seconds) to wait for the expected state to be reached. The value of
            ``None`` (the default) means no client-side timeout.
        :param abort_on_failed_state: If ``True`` (the default), abort waiting
            if any node reaches a failure state which does not match the
            expected one. Note that the failure state for ``enroll`` ->
            ``manageable`` transition is ``enroll`` again.
        :param fail: If set to ``False`` this call will not raise on timeouts
            and provisioning failures.

        :return: If `fail` is ``True`` (the default), the list of
            :class:`~openstack.baremetal.v1.node.Node` instances that reached
            the requested state. If `fail` is ``False``, a
            :class:`~openstack.baremetal.v1.node.WaitResult` named tuple.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if a node
            reaches an error state and ``abort_on_failed_state`` is ``True``.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        """
    log_nodes = ', '.join((n.id if isinstance(n, _node.Node) else n for n in nodes))
    finished = []
    failed = []
    remaining = nodes
    try:
        for count in utils.iterate_timeout(timeout, "Timeout waiting for nodes %(nodes)s to reach target state '%(state)s'" % {'nodes': log_nodes, 'state': expected_state}):
            nodes = [self.get_node(n) for n in remaining]
            remaining = []
            for n in nodes:
                try:
                    if n._check_state_reached(self, expected_state, abort_on_failed_state):
                        finished.append(n)
                    else:
                        remaining.append(n)
                except exceptions.ResourceFailure:
                    if fail:
                        raise
                    else:
                        failed.append(n)
            if not remaining:
                if fail:
                    return finished
                else:
                    return _node.WaitResult(finished, failed, [])
            self.log.debug('Still waiting for nodes %(nodes)s to reach state "%(target)s"', {'nodes': ', '.join((n.id for n in remaining)), 'target': expected_state})
    except exceptions.ResourceTimeout:
        if fail:
            raise
        else:
            return _node.WaitResult(finished, failed, remaining)