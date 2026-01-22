from urllib import parse
from saharaclient.api import base
class ClusterManagerV2(ClusterManagerV1):

    def create(self, name, plugin_name, plugin_version, cluster_template_id=None, default_image_id=None, is_transient=None, description=None, cluster_configs=None, node_groups=None, user_keypair_id=None, anti_affinity=None, net_id=None, count=None, use_autoconfig=None, shares=None, is_public=None, is_protected=None):
        """Launch a Cluster."""
        data = {'name': name, 'plugin_name': plugin_name, 'plugin_version': plugin_version}
        return self._do_create(data, cluster_template_id, default_image_id, is_transient, description, cluster_configs, node_groups, user_keypair_id, anti_affinity, net_id, count, use_autoconfig, shares, is_public, is_protected, api_ver=2)

    def scale(self, cluster_id, scale_object):
        """Scale an existing Cluster.

        :param scale_object: dict that describes scaling operation

        :Example:

        The following `scale_object` can be used to change the number of
        instances in the node group (optionally specifiying which instances to
        delete) or add instances of a new node group to an existing cluster:

        .. sourcecode:: json

            {
                "add_node_groups": [
                    {
                        "count": 3,
                        "name": "new_ng",
                        "node_group_template_id": "ngt_id"
                    }
                ],
                "resize_node_groups": [
                    {
                        "count": 2,
                        "name": "old_ng",
                        "instances": ["instance_id1", "instance_id2"]
                    }
                ]
            }

        """
        return self._update('/clusters/%s' % cluster_id, scale_object)

    def force_delete(self, cluster_id):
        """Force Delete a Cluster."""
        data = {'force': True}
        return self._delete('/clusters/%s' % cluster_id, data)

    def update_keypair(self, cluster_id):
        """Reflect an updated keypair on the cluster."""
        data = {'update_keypair': True}
        return self._patch('/clusters/%s' % cluster_id, data)