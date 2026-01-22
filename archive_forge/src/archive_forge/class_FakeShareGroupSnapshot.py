import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareGroupSnapshot(object):
    """Fake a share group snapshot"""

    @staticmethod
    def create_one_share_group_snapshot(attrs=None, methods=None):
        """Create a fake share group snapshot

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        methods = methods or {}
        share_group_snapshot = {'status': 'available', 'share_group_id': 'share-group-id-' + uuid.uuid4().hex, 'name': None, 'created_at': datetime.datetime.now().isoformat(), 'project_id': 'project-id-' + uuid.uuid4().hex, 'id': 'share-group-snapshot-id-' + uuid.uuid4().hex, 'description': None}
        share_group_snapshot.update(attrs)
        share_group_snapshot = osc_fakes.FakeResource(info=copy.deepcopy(share_group_snapshot), methods=methods, loaded=True)
        return share_group_snapshot

    @staticmethod
    def create_share_group_snapshots(attrs=None, count=2):
        """Create multiple fake share group snapshot.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share group snapshot to be faked
        :return:
            A list of FakeResource objects
        """
        share_group_snapshots = []
        for n in range(0, count):
            share_group_snapshots.append(FakeShareGroupSnapshot.create_one_share_group_snapshot(attrs))
        return share_group_snapshots