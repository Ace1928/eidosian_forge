from openstack import _log
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
class Introspection(resource.Resource):
    resources_key = 'introspection'
    base_path = '/introspection'
    allow_create = True
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    create_method = 'POST'
    create_requires_id = True
    create_returns_body = False
    finished_at = resource.Body('finished_at')
    error = resource.Body('error')
    id = resource.Body('uuid', alternate_id=True)
    is_finished = resource.Body('finished', type=bool)
    links = resource.Body('links', type=list)
    started_at = resource.Body('started_at')
    state = resource.Body('state')

    def abort(self, session):
        """Abort introspection.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        """
        if self.is_finished:
            return
        session = self._get_session(session)
        version = self._get_microversion(session, action='delete')
        request = self._prepare_request(requires_id=True)
        request.url = utils.urljoin(request.url, 'abort')
        response = session.post(request.url, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
        msg = 'Failed to abort introspection for node {id}'.format(id=self.id)
        exceptions.raise_from_response(response, error_message=msg)

    def get_data(self, session, processed=True):
        """Get introspection data.

        Note that the introspection data format is not stable and can vary
        from environment to environment.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param processed: Whether to fetch the final processed data (the
            default) or the raw unprocessed data as received from the ramdisk.
        :type processed: bool
        :returns: introspection data from the most recent successful run.
        :rtype: dict
        """
        session = self._get_session(session)
        version = self._get_microversion(session, action='fetch') if processed else '1.17'
        request = self._prepare_request(requires_id=True)
        request.url = utils.urljoin(request.url, 'data')
        if not processed:
            request.url = utils.urljoin(request.url, 'unprocessed')
        response = session.get(request.url, headers=request.headers, microversion=version)
        msg = 'Failed to fetch introspection data for node {id}'.format(id=self.id)
        exceptions.raise_from_response(response, error_message=msg)
        return response.json()

    def wait(self, session, timeout=None, ignore_error=False):
        """Wait for the node to reach the expected state.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param timeout: How much (in seconds) to wait for the introspection.
            The value of ``None`` (the default) means no client-side timeout.
        :param ignore_error: If ``True``, this call will raise an exception
            if the introspection reaches the ``error`` state. Otherwise the
            error state is considered successful and the call returns.
        :return: This :class:`Introspection` instance.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if
            introspection fails and ``ignore_error`` is ``False``.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        """
        if self._check_state(ignore_error):
            return self
        for count in utils.iterate_timeout(timeout, 'Timeout waiting for introspection on node %s' % self.id):
            self.fetch(session)
            if self._check_state(ignore_error):
                return self
            _logger.debug('Still waiting for introspection of node %(node)s, the current state is "%(state)s"', {'node': self.id, 'state': self.state})

    def _check_state(self, ignore_error):
        if self.state == 'error' and (not ignore_error):
            raise exceptions.ResourceFailure('Introspection of node %(node)s failed: %(error)s' % {'node': self.id, 'error': self.error})
        else:
            return self.is_finished