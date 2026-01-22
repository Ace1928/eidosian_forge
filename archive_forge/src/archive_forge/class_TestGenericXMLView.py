import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
class TestGenericXMLView(base.BaseTestCase):

    def setUp(self):
        super(TestGenericXMLView, self).setUp()
        self.model = mwdv_generator()
        self.model.set_current_view_type('xml')

    def test_dict_serialization(self):
        self.model['dt'] = {'a': 1, 'b': 2}
        target_str = '<model><dt><a>1</a><b>2</b></dt><int>1</int><string>value</string></model>'
        self.assertEqual(target_str, str(self.model))

    def test_list_serialization(self):
        self.model['lt'] = ['a', 'b']
        target_str = '<model><int>1</int><lt><item>a</item><item>b</item></lt><string>value</string></model>'
        self.assertEqual(target_str, str(self.model))

    def test_list_in_dict_serialization(self):
        self.model['dt'] = {'a': 1, 'b': [2, 3]}
        target_str = '<model><dt><a>1</a><b><item>2</item><item>3</item></b></dt><int>1</int><string>value</string></model>'
        self.assertEqual(target_str, str(self.model))

    def test_dict_in_list_serialization(self):
        self.model['lt'] = [1, {'b': 2, 'c': 3}]
        target_str = '<model><int>1</int><lt><item>1</item><item><b>2</b><c>3</c></item></lt><string>value</string></model>'
        self.assertEqual(target_str, str(self.model))

    def test_submodel_serialization(self):
        sm = mwdv_generator()
        sm.set_current_view_type('xml')
        self.model['submodel'] = sm
        target_str = '<model><int>1</int><string>value</string><submodel><model><int>1</int><string>value</string></model></submodel></model>'
        self.assertEqual(target_str, str(self.model))

    def test_wrapper_name(self):
        self.model.attached_view.wrapper_name = 'cheese'
        target_str = '<cheese><int>1</int><string>value</string></cheese>'
        self.assertEqual(target_str, str(self.model))