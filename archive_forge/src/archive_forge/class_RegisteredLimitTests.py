import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
class RegisteredLimitTests(object):

    def test_create_registered_limit_crud(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex, description='test description')
        reg_limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
        self.assertDictEqual(registered_limit_1, reg_limits[0])
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
        registered_limit_3 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='backup', default_limit=5, id=uuid.uuid4().hex)
        reg_limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_2, registered_limit_3])
        self.assertEqual(2, len(reg_limits))
        for reg_limit in reg_limits:
            if reg_limit['id'] == registered_limit_2['id']:
                self.assertDictEqual(registered_limit_2, reg_limit)
            if reg_limit['id'] == registered_limit_3['id']:
                self.assertDictEqual(registered_limit_3, reg_limit)

    def test_create_registered_limit_duplicate(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        self.assertRaises(exception.Conflict, PROVIDERS.unified_limit_api.create_registered_limits, [registered_limit_2])

    def test_create_multi_registered_limits_duplicate(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_3 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=10, id=uuid.uuid4().hex)
        self.assertRaises(exception.Conflict, PROVIDERS.unified_limit_api.create_registered_limits, [registered_limit_2, registered_limit_3])
        reg_limits = PROVIDERS.unified_limit_api.list_registered_limits()
        self.assertEqual(1, len(reg_limits))
        self.assertEqual(registered_limit_1['id'], reg_limits[0]['id'])

    def test_create_registered_limit_invalid_service(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=uuid.uuid4().hex, region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        self.assertRaises(exception.ValidationError, PROVIDERS.unified_limit_api.create_registered_limits, [registered_limit_1])

    def test_create_registered_limit_invalid_region(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=uuid.uuid4().hex, resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        self.assertRaises(exception.ValidationError, PROVIDERS.unified_limit_api.create_registered_limits, [registered_limit_1])

    def test_create_registered_limit_description_none(self):
        registered_limit = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex, description=None)
        res = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
        self.assertIsNone(res[0]['description'])

    def test_create_registered_limit_without_description(self):
        registered_limit = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit.pop('description')
        res = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
        self.assertIsNone(res[0]['description'])

    def test_update_registered_limit(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
        expect_region = 'region_two'
        registered_limit_update = {'id': registered_limit_1['id'], 'region_id': expect_region}
        res = PROVIDERS.unified_limit_api.update_registered_limit(registered_limit_1['id'], registered_limit_update)
        self.assertEqual(expect_region, res['region_id'])
        registered_limit_update = {'region_id': expect_region}
        res = PROVIDERS.unified_limit_api.update_registered_limit(registered_limit_2['id'], registered_limit_update)
        self.assertEqual(expect_region, res['region_id'])

    def test_update_registered_limit_invalid_input_return_bad_request(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
        update_ref = {'id': registered_limit_1['id'], 'service_id': uuid.uuid4().hex}
        self.assertRaises(exception.ValidationError, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_1['id'], update_ref)
        update_ref = {'id': registered_limit_1['id'], 'region_id': 'fake_id'}
        self.assertRaises(exception.ValidationError, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_1['id'], update_ref)

    def test_update_registered_limit_duplicate(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
        update_ref = {'id': registered_limit_1['id'], 'region_id': self.region_two['id'], 'resource_name': 'snapshot'}
        self.assertRaises(exception.Conflict, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_1['id'], update_ref)

    def test_update_registered_limit_when_reference_limit_exist(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
        limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_limits([limit_1])
        registered_limit_update = {'id': registered_limit_1['id'], 'region_id': 'region_two'}
        self.assertRaises(exception.RegisteredLimitError, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_1['id'], registered_limit_update)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_2])
        limit_2 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_limits([limit_2])
        registered_limit_update = {'id': registered_limit_2['id'], 'region_id': 'region_two'}
        self.assertRaises(exception.RegisteredLimitError, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_2['id'], registered_limit_update)

    def test_list_registered_limits(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
        reg_limits_1 = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
        reg_limits_2 = PROVIDERS.unified_limit_api.list_registered_limits()
        self.assertEqual(2, len(reg_limits_2))
        self.assertDictEqual(reg_limits_1[0], reg_limits_2[0])
        self.assertDictEqual(reg_limits_1[1], reg_limits_2[1])

    def test_list_registered_limit_by_limit(self):
        self.config_fixture.config(list_limit=1)
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
        hints = driver_hints.Hints()
        reg_limits = PROVIDERS.unified_limit_api.list_registered_limits(hints=hints)
        self.assertEqual(1, len(reg_limits))
        if reg_limits[0]['id'] == registered_limit_1['id']:
            self.assertDictEqual(registered_limit_1, reg_limits[0])
        else:
            self.assertDictEqual(registered_limit_2, reg_limits[0])

    def test_list_registered_limit_by_filter(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
        hints = driver_hints.Hints()
        hints.add_filter('service_id', self.service_one['id'])
        res = PROVIDERS.unified_limit_api.list_registered_limits(hints)
        self.assertEqual(2, len(res))
        hints = driver_hints.Hints()
        hints.add_filter('region_id', self.region_one['id'])
        res = PROVIDERS.unified_limit_api.list_registered_limits(hints)
        self.assertEqual(1, len(res))
        hints = driver_hints.Hints()
        hints.add_filter('resource_name', 'backup')
        res = PROVIDERS.unified_limit_api.list_registered_limits(hints)
        self.assertEqual(0, len(res))

    def test_get_registered_limit(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
        res = PROVIDERS.unified_limit_api.get_registered_limit(registered_limit_2['id'])
        self.assertDictEqual(registered_limit_2, res)

    def test_get_registered_limit_returns_not_found(self):
        self.assertRaises(exception.RegisteredLimitNotFound, PROVIDERS.unified_limit_api.get_registered_limit, uuid.uuid4().hex)

    def test_delete_registered_limit(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
        PROVIDERS.unified_limit_api.delete_registered_limit(registered_limit_1['id'])
        self.assertRaises(exception.RegisteredLimitNotFound, PROVIDERS.unified_limit_api.get_registered_limit, registered_limit_1['id'])
        reg_limits = PROVIDERS.unified_limit_api.list_registered_limits()
        self.assertEqual(1, len(reg_limits))
        self.assertEqual(registered_limit_2['id'], reg_limits[0]['id'])

    def test_delete_registered_limit_returns_not_found(self):
        self.assertRaises(exception.RegisteredLimitNotFound, PROVIDERS.unified_limit_api.delete_registered_limit, uuid.uuid4().hex)

    def test_delete_registered_limit_when_reference_limit_exist(self):
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
        limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_limits([limit_1])
        self.assertRaises(exception.RegisteredLimitError, PROVIDERS.unified_limit_api.delete_registered_limit, registered_limit_1['id'])
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_2])
        limit_2 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_limits([limit_2])
        self.assertRaises(exception.RegisteredLimitError, PROVIDERS.unified_limit_api.delete_registered_limit, registered_limit_2['id'])