from mistralclient.api import base
class MemberManager(base.ResourceManager):
    resource_class = Member

    def create(self, resource_id, resource_type, member_id):
        self._ensure_not_empty(resource_id=resource_id, resource_type=resource_type, member_id=member_id)
        data = {'member_id': member_id}
        url = '/%ss/%s/members' % (resource_type, resource_id)
        return self._create(url, data)

    def update(self, resource_id, resource_type, member_id='', status='accepted'):
        if not member_id:
            member_id = self.http_client.project_id
        url = '/%ss/%s/members/%s' % (resource_type, resource_id, member_id)
        return self._update(url, {'status': status})

    def list(self, resource_id, resource_type):
        url = '/%ss/%s/members' % (resource_type, resource_id)
        return self._list(url, response_key='members')

    def get(self, resource_id, resource_type, member_id=None):
        self._ensure_not_empty(resource_id=resource_id, resource_type=resource_type)
        if not member_id:
            member_id = self.http_client.project_id
        url = '/%ss/%s/members/%s' % (resource_type, resource_id, member_id)
        return self._get(url)

    def delete(self, resource_id, resource_type, member_id):
        self._ensure_not_empty(resource_id=resource_id, resource_type=resource_type, member_id=member_id)
        url = '/%ss/%s/members/%s' % (resource_type, resource_id, member_id)
        self._delete(url)