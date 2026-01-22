import requests
import uuid
from urllib import parse as urlparse
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
from keystoneclient.v3 import client
class CrudTests(object):
    key = None
    collection_key = None
    model = None
    manager = None
    path_prefix = None

    def new_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        kwargs.setdefault(uuid.uuid4().hex, uuid.uuid4().hex)
        return kwargs

    def encode(self, entity):
        if isinstance(entity, dict):
            return {self.key: entity}
        if isinstance(entity, list):
            return {self.collection_key: entity}
        raise NotImplementedError('Are you sure you want to encode that?')

    def stub_entity(self, method, parts=None, entity=None, id=None, **kwargs):
        if entity:
            entity = self.encode(entity)
            kwargs['json'] = entity
        if not parts:
            parts = [self.collection_key]
            if self.path_prefix:
                parts.insert(0, self.path_prefix)
        if id:
            if not parts:
                parts = []
            parts.append(id)
        self.stub_url(method, parts=parts, **kwargs)

    def assertEntityRequestBodyIs(self, entity):
        self.assertRequestBodyIs(json=self.encode(entity))

    def test_create(self, ref=None, req_ref=None):
        deprecations = self.useFixture(client_fixtures.Deprecations())
        deprecations.expect_deprecations()
        ref = ref or self.new_ref()
        manager_ref = ref.copy()
        manager_ref.pop('id')
        if req_ref:
            req_ref = req_ref.copy()
        else:
            req_ref = ref.copy()
            req_ref.pop('id')
        self.stub_entity('POST', entity=req_ref, status_code=201)
        returned = self.manager.create(**parameterize(manager_ref))
        self.assertIsInstance(returned, self.model)
        for attr in req_ref:
            self.assertEqual(getattr(returned, attr), req_ref[attr], 'Expected different %s' % attr)
        self.assertEntityRequestBodyIs(req_ref)
        return returned

    def test_get(self, ref=None):
        ref = ref or self.new_ref()
        self.stub_entity('GET', id=ref['id'], entity=ref)
        returned = self.manager.get(ref['id'])
        self.assertIsInstance(returned, self.model)
        for attr in ref:
            self.assertEqual(getattr(returned, attr), ref[attr], 'Expected different %s' % attr)

    def _get_expected_path(self, expected_path=None):
        if not expected_path:
            if self.path_prefix:
                expected_path = 'v3/%s/%s' % (self.path_prefix, self.collection_key)
            else:
                expected_path = 'v3/%s' % self.collection_key
        return expected_path

    def test_list_by_id(self, ref=None, **filter_kwargs):
        """Test ``entities.list(id=x)`` being rewritten as ``GET /v3/entities/x``.  # noqa

        This tests an edge case of each manager's list() implementation, to
        ensure that it "does the right thing" when users call ``.list()``
        when they should have used ``.get()``.

        """
        if 'id' not in filter_kwargs:
            ref = ref or self.new_ref()
            filter_kwargs['id'] = ref['id']
        self.assertRaises(TypeError, self.manager.list, **filter_kwargs)

    def test_list(self, ref_list=None, expected_path=None, expected_query=None, **filter_kwargs):
        ref_list = ref_list or [self.new_ref(), self.new_ref()]
        expected_path = self._get_expected_path(expected_path)
        self.requests_mock.get(urlparse.urljoin(self.TEST_URL, expected_path), json=self.encode(ref_list))
        returned_list = self.manager.list(**filter_kwargs)
        self.assertEqual(len(ref_list), len(returned_list))
        [self.assertIsInstance(r, self.model) for r in returned_list]
        qs_args = self.requests_mock.last_request.qs
        qs_args_expected = expected_query or filter_kwargs
        for key, value in qs_args_expected.items():
            self.assertIn(key, qs_args)
            self.assertIn(str(value).lower(), qs_args[key])
        for key in qs_args:
            self.assertIn(key, qs_args_expected)

    def test_list_params(self):
        ref_list = [self.new_ref()]
        filter_kwargs = {uuid.uuid4().hex: uuid.uuid4().hex}
        expected_path = self._get_expected_path()
        self.requests_mock.get(urlparse.urljoin(self.TEST_URL, expected_path), json=self.encode(ref_list))
        self.manager.list(**filter_kwargs)
        self.assertQueryStringContains(**filter_kwargs)

    def test_find(self, ref=None):
        ref = ref or self.new_ref()
        ref_list = [ref]
        self.stub_entity('GET', entity=ref_list)
        returned = self.manager.find(name=getattr(ref, 'name', None))
        self.assertIsInstance(returned, self.model)
        for attr in ref:
            self.assertEqual(getattr(returned, attr), ref[attr], 'Expected different %s' % attr)
        if hasattr(ref, 'name'):
            self.assertQueryStringIs('name=%s' % ref['name'])
        else:
            self.assertQueryStringIs('')

    def test_update(self, ref=None, req_ref=None):
        deprecations = self.useFixture(client_fixtures.Deprecations())
        deprecations.expect_deprecations()
        ref = ref or self.new_ref()
        self.stub_entity('PATCH', id=ref['id'], entity=ref)
        if req_ref:
            req_ref = req_ref.copy()
        else:
            req_ref = ref.copy()
            req_ref.pop('id')
        returned = self.manager.update(ref['id'], **parameterize(req_ref))
        self.assertIsInstance(returned, self.model)
        for attr in ref:
            self.assertEqual(getattr(returned, attr), ref[attr], 'Expected different %s' % attr)
        self.assertEntityRequestBodyIs(req_ref)

    def test_delete(self, ref=None):
        ref = ref or self.new_ref()
        self.stub_entity('DELETE', id=ref['id'], status_code=204)
        self.manager.delete(ref['id'])