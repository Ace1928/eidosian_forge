import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeNodeGroup(object):
    """Fake one or more NodeGroup."""

    @staticmethod
    def create_one_nodegroup(attrs=None):
        """Create a fake NodeGroup.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with flavor_id, image_id, and so on
        """
        attrs = attrs or {}
        nodegroup_info = {'created_at': '2017-03-16T18:40:39+00:00', 'updated_at': '2017-03-16T18:40:45+00:00', 'uuid': '3a369884-b6ba-484f-a206-919b4b718aff', 'cluster_id': 'fake-cluster', 'docker_volume_size': None, 'node_addresses': [], 'labels': {}, 'labels_overridden': {}, 'labels_added': {}, 'labels_skipped': {}, 'node_count': 1, 'name': 'fake-nodegroup', 'flavor_id': 'm1.medium', 'image_id': 'fedora-latest', 'project_id': None, 'role': 'worker', 'max_node_count': 10, 'min_node_count': 1, 'is_default': False, 'stack_id': '3a369884-b6ba-484f-fake-stackb718aff', 'status': 'CREATE_COMPLETE', 'status_reason': 'None', 'master_lb_enabled': False}
        nodegroup_info.update(attrs)
        nodegroup = osc_fakes.FakeResource(info=copy.deepcopy(nodegroup_info), loaded=True)
        return nodegroup