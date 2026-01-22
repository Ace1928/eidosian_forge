import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
class TestTagController(testtools.TestCase):

    def setUp(self):
        super(TestTagController, self).setUp()
        self.api = utils.FakeAPI(data_fixtures)
        self.schema_api = utils.FakeSchemaAPI(schema_fixtures)
        self.controller = base.BaseController(self.api, self.schema_api, metadefs.TagController)

    def test_list_tag(self):
        tags = self.controller.list(NAMESPACE1)
        actual = [tag.name for tag in tags]
        self.assertEqual([TAG1, TAG2], actual)

    def test_get_tag(self):
        tag = self.controller.get(NAMESPACE1, TAG1)
        self.assertEqual(TAG1, tag.name)

    def test_create_tag(self):
        tag = self.controller.create(NAMESPACE1, TAGNEW1)
        self.assertEqual(TAGNEW1, tag.name)

    def test_create_multiple_tags(self):
        properties = {'tags': [TAGNEW2, TAGNEW3]}
        tags = self.controller.create_multiple(NAMESPACE1, **properties)
        self.assertEqual([TAGNEW2, TAGNEW3], tags)

    def test_update_tag(self):
        properties = {'name': TAG2}
        tag = self.controller.update(NAMESPACE1, TAG1, **properties)
        self.assertEqual(TAG2, tag.name)

    def test_delete_tag(self):
        self.controller.delete(NAMESPACE1, TAG1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s/tags/%s' % (NAMESPACE1, TAG1), {}, None)]
        self.assertEqual(expect, self.api.calls)

    def test_delete_all_tags(self):
        self.controller.delete_all(NAMESPACE1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s/tags' % NAMESPACE1, {}, None)]
        self.assertEqual(expect, self.api.calls)