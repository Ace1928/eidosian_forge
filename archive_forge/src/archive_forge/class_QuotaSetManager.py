from cinderclient import base
class QuotaSetManager(base.Manager):
    resource_class = QuotaSet

    def get(self, tenant_id, usage=False):
        if hasattr(tenant_id, 'tenant_id'):
            tenant_id = tenant_id.tenant_id
        return self._get('/os-quota-sets/%s?usage=%s' % (tenant_id, usage), 'quota_set')

    def update(self, tenant_id, **updates):
        skip_validation = updates.pop('skip_validation', True)
        body = {'quota_set': {'tenant_id': tenant_id}}
        for update in updates:
            body['quota_set'][update] = updates[update]
        request_url = '/os-quota-sets/%s' % tenant_id
        if not skip_validation:
            request_url += '?skip_validation=False'
        result = self._update(request_url, body)
        return self.resource_class(self, result['quota_set'], loaded=True, resp=result.request_ids)

    def defaults(self, tenant_id):
        return self._get('/os-quota-sets/%s/defaults' % tenant_id, 'quota_set')

    def delete(self, tenant_id):
        if hasattr(tenant_id, 'tenant_id'):
            tenant_id = tenant_id.tenant_id
        return self._delete('/os-quota-sets/%s' % tenant_id)