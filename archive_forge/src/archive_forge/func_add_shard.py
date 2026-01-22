from troveclient import base
from troveclient import common
def add_shard(self, cluster):
    """Adds a shard to the specified cluster.

        :param cluster: The cluster to add a shard to
        """
    url = '/clusters/%s' % base.getid(cluster)
    body = {'add_shard': {}}
    resp, body = self.api.client.post(url, body=body)
    common.check_for_exceptions(resp, body, url)
    if body:
        return self.resource_class(self, body, loaded=True)
    return body