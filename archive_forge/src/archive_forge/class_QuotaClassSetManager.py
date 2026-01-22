from cinderclient import base
class QuotaClassSetManager(base.Manager):
    resource_class = QuotaClassSet

    def get(self, class_name):
        return self._get('/os-quota-class-sets/%s' % class_name, 'quota_class_set')

    def update(self, class_name, **updates):
        quota_class_set = {}
        for update in updates:
            quota_class_set[update] = updates[update]
        result = self._update('/os-quota-class-sets/%s' % class_name, {'quota_class_set': quota_class_set})
        return self.resource_class(self, result['quota_class_set'], loaded=True, resp=result.request_ids)