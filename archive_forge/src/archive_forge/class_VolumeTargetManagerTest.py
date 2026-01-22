import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeTargetManagerTest(VolumeTargetManagerTestBase):

    def setUp(self):
        super(VolumeTargetManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = ironicclient.v1.volume_target.VolumeTargetManager(self.api)

    def test_volume_targets_list(self):
        volume_targets = self.mgr.list()
        expect = [('GET', '/v1/volume/targets', {}, None)]
        expect_targets = [TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_targets_list_by_node(self):
        volume_targets = self.mgr.list(node=NODE_UUID)
        expect = [('GET', '/v1/volume/targets/?node=%s' % NODE_UUID, {}, None)]
        expect_targets = [TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_targets_list_by_node_detail(self):
        volume_targets = self.mgr.list(node=NODE_UUID, detail=True)
        expect = [('GET', '/v1/volume/targets/?detail=True&node=%s' % NODE_UUID, {}, None)]
        expect_targets = [TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_targets_list_detail(self):
        volume_targets = self.mgr.list(detail=True)
        expect = [('GET', '/v1/volume/targets/?detail=True', {}, None)]
        expect_targets = [TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_target_list_fields(self):
        volume_targets = self.mgr.list(fields=['uuid', 'boot_index'])
        expect = [('GET', '/v1/volume/targets/?fields=uuid,boot_index', {}, None)]
        expect_targets = [TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_target_list_detail_and_fields_fail(self):
        self.assertRaises(exc.InvalidAttribute, self.mgr.list, detail=True, fields=['uuid', 'boot_index'])

    def test_volume_targets_show(self):
        volume_target = self.mgr.get(TARGET1['uuid'])
        expect = [('GET', '/v1/volume/targets/%s' % TARGET1['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(TARGET1, volume_target)

    def test_volume_target_show_fields(self):
        volume_target = self.mgr.get(TARGET1['uuid'], fields=['uuid', 'boot_index'])
        expect = [('GET', '/v1/volume/targets/%s?fields=uuid,boot_index' % TARGET1['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(TARGET1['uuid'], volume_target.uuid)
        self.assertEqual(TARGET1['boot_index'], volume_target.boot_index)

    def test_create(self):
        volume_target = self.mgr.create(**CREATE_TARGET)
        expect = [('POST', '/v1/volume/targets', {}, CREATE_TARGET)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(TARGET1, volume_target)

    def test_create_with_uuid(self):
        volume_target = self.mgr.create(**CREATE_TARGET_WITH_UUID)
        expect = [('POST', '/v1/volume/targets', {}, CREATE_TARGET_WITH_UUID)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(TARGET1, volume_target)

    def test_delete(self):
        volume_target = self.mgr.delete(TARGET1['uuid'])
        expect = [('DELETE', '/v1/volume/targets/%s' % TARGET1['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(volume_target)

    def test_update(self):
        patch = {'op': 'replace', 'value': NEW_VALUE, 'path': '/boot_index'}
        volume_target = self.mgr.update(volume_target_id=TARGET1['uuid'], patch=patch)
        expect = [('PATCH', '/v1/volume/targets/%s' % TARGET1['uuid'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(UPDATED_TARGET, volume_target)