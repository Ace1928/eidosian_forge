from magnumclient.v1 import baseunit
class ClusterManager(baseunit.BaseTemplateManager):
    resource_class = Cluster
    template_name = 'clusters'

    def resize(self, cluster_uuid, node_count, nodes_to_remove=[], nodegroup=None):
        url = self._path(cluster_uuid) + '/actions/resize'
        post_body = {'node_count': node_count}
        if nodes_to_remove:
            post_body.update({'nodes_to_remove': nodes_to_remove})
        if nodegroup:
            post_body.update({'nodegroup': nodegroup})
        resp, resp_body = self.api.json_request('POST', url, body=post_body)
        if resp_body:
            return self.resource_class(self, resp_body)

    def upgrade(self, cluster_uuid, cluster_template, max_batch_size=1, nodegroup=None):
        url = self._path(cluster_uuid) + '/actions/upgrade'
        post_body = {'cluster_template': cluster_template}
        if max_batch_size:
            post_body.update({'max_batch_size': max_batch_size})
        if nodegroup:
            post_body.update({'nodegroup': nodegroup})
        resp, resp_body = self.api.json_request('POST', url, body=post_body)
        if resp_body:
            return self.resource_class(self, resp_body)