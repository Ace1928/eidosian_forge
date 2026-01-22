from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
class WorkerManager(base.Manager):
    base_url = '/workers'

    @api_versions.wraps('3.24')
    def clean(self, **filters):
        url = self.base_url + '/cleanup'
        resp, body = self.api.client.post(url, body=filters)
        cleaning = Service.list_factory(self, body['cleaning'])
        unavailable = Service.list_factory(self, body['unavailable'])
        result = common_base.TupleWithMeta((cleaning, unavailable), resp)
        return result