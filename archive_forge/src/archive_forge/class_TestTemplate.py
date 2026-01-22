from unittest import mock
from oslotest import base
from vitrageclient import exceptions as exc
from vitrageclient.tests.utils import get_resources_dir
from vitrageclient.v1.template import Template
class TestTemplate(base.BaseTestCase):

    def test_validate_by_path(self):
        template_path = get_resources_dir() + '/template1.yaml'
        template = Template(mock.Mock())
        template.validate(path=template_path)

    def test_validate_by_nonexisting_path(self):
        template = Template(mock.Mock())
        self.assertRaises(IOError, template.validate, path='non_existing_template_path.yaml')

    def test_validate_by_template(self):
        template = Template(mock.Mock())
        template.validate(template_str=TEMPLATE_STRING)

    def test_validate_by_nothing(self):
        template = Template(mock.Mock())
        self.assertRaises(exc.CommandError, template.validate)

    def test_add_by_path(self):
        template_path = get_resources_dir() + '/template1.yaml'
        template = Template(mock.Mock())
        template.add(path=template_path)

    def test_add_by_nonexisting_path(self):
        template = Template(mock.Mock())
        self.assertRaises(IOError, template.add, path='non_existing_template_path.yaml')

    def test_add_by_template(self):
        template = Template(mock.Mock())
        template.add(template_str=TEMPLATE_STRING)

    def test_add_by_nothing(self):
        template = Template(mock.Mock())
        self.assertRaises(exc.CommandError, template.add)