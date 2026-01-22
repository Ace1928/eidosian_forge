from openstack import exceptions
from openstack import resource
from openstack import utils
class ShareAccessRule(resource.Resource):
    resource_key = 'access'
    resources_key = 'access_list'
    base_path = '/share-access-rules'
    allow_create = True
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    allow_head = False
    _query_mapping = resource.QueryParameters('share_id')
    _max_microversion = '2.82'
    access_key = resource.Body('access_key', type=str)
    access_level = resource.Body('access_level', type=str)
    access_list = resource.Body('access_list', type=str)
    access_to = resource.Body('access_to', type=str)
    access_type = resource.Body('access_type', type=str)
    created_at = resource.Body('created_at', type=str)
    metadata = resource.Body('metadata', type=dict)
    share_id = resource.Body('share_id', type=str)
    state = resource.Body('state', type=str)
    updated_at = resource.Body('updated_at', type=str)
    lock_visibility = resource.Body('lock_visibility', type=bool)
    lock_deletion = resource.Body('lock_deletion', type=bool)
    lock_reason = resource.Body('lock_reason', type=bool)

    def _action(self, session, body, url, action='patch', microversion=None):
        headers = {'Accept': ''}
        if microversion is None:
            microversion = self._get_microversion(session, action=action)
        return session.post(url, json=body, headers=headers, microversion=microversion)

    def create(self, session, **kwargs):
        return super().create(session, resource_request_key='allow_access', resource_response_key='access', **kwargs)

    def delete(self, session, share_id, ignore_missing=True, *, unrestrict=False):
        body = {'deny_access': {'access_id': self.id}}
        if unrestrict:
            body['deny_access']['unrestrict'] = True
        url = utils.urljoin('/shares', share_id, 'action')
        response = self._action(session, body, url)
        try:
            response = self._action(session, body, url)
            self._translate_response(response)
        except exceptions.ResourceNotFound:
            if not ignore_missing:
                raise
        return response