from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
class Allocation(_common.Resource):
    resources_key = 'allocations'
    base_path = '/allocations'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_patch = True
    commit_method = 'PATCH'
    commit_jsonpatch = True
    _query_mapping = resource.QueryParameters('node', 'resource_class', 'state', 'owner', fields={'type': _common.fields_type})
    _max_microversion = '1.60'
    candidate_nodes = resource.Body('candidate_nodes', type=list)
    created_at = resource.Body('created_at')
    extra = resource.Body('extra', type=dict)
    id = resource.Body('uuid', alternate_id=True)
    last_error = resource.Body('last_error')
    links = resource.Body('links', type=list)
    name = resource.Body('name')
    node = resource.Body('node')
    node_id = resource.Body('node_uuid')
    owner = resource.Body('owner')
    resource_class = resource.Body('resource_class')
    state = resource.Body('state')
    traits = resource.Body('traits', type=list)
    updated_at = resource.Body('updated_at')

    def wait(self, session, timeout=None, ignore_error=False):
        """Wait for the allocation to become active.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param timeout: How much (in seconds) to wait for the allocation.
            The value of ``None`` (the default) means no client-side timeout.
        :param ignore_error: If ``True``, this call will raise an exception
            if the allocation reaches the ``error`` state. Otherwise the error
            state is considered successful and the call returns.

        :return: This :class:`Allocation` instance.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if allocation
            fails and ``ignore_error`` is ``False``.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        """
        if self.state == 'active':
            return self
        for count in utils.iterate_timeout(timeout, 'Timeout waiting for the allocation %s' % self.id):
            self.fetch(session)
            if self.state == 'error' and (not ignore_error):
                raise exceptions.ResourceFailure('Allocation %(allocation)s failed: %(error)s' % {'allocation': self.id, 'error': self.last_error})
            elif self.state != 'allocating':
                return self
            session.log.debug('Still waiting for the allocation %(allocation)s to become active, the current state is %(state)s', {'allocation': self.id, 'state': self.state})