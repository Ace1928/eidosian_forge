import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def _cluster_templates_configure_ng(app, node_groups, client):
    node_groups_list = dict(map(lambda x: x.split(':', 1), node_groups))
    node_groups = []
    plugins_versions = set()
    for name, count in node_groups_list.items():
        ng = get_resource(client.node_group_templates, name)
        node_groups.append({'name': ng.name, 'count': int(count), 'node_group_template_id': ng.id})
        if is_api_v2(app):
            plugins_versions.add((ng.plugin_name, ng.plugin_version))
        else:
            plugins_versions.add((ng.plugin_name, ng.hadoop_version))
    if len(plugins_versions) != 1:
        raise exceptions.CommandError('Node groups with the same plugins and versions must be specified')
    plugin, plugin_version = plugins_versions.pop()
    return (plugin, plugin_version, node_groups)