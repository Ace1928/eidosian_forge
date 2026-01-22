import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
class TestModelReportType(base.BaseTestCase):

    def test_model_with_default_views(self):
        model = mwdv_generator()
        model.set_current_view_type('text')
        self.assertEqual('int = 1\nstring = value', str(model))
        model.set_current_view_type('json')
        self.assertEqual('{"int": 1, "string": "value"}', str(model))
        model.set_current_view_type('xml')
        self.assertEqual('<model><int>1</int><string>value</string></model>', str(model))

    def test_recursive_type_propagation_with_nested_models(self):
        model = mwdv_generator()
        model['submodel'] = mwdv_generator()
        model.set_current_view_type('json')
        self.assertEqual(model.submodel.views['json'], model.submodel.attached_view)

    def test_recursive_type_propagation_with_nested_dicts(self):
        nested_model = mwdv.ModelWithDefaultViews(json_view='abc')
        data = {'a': 1, 'b': {'c': nested_model}}
        top_model = base_model.ReportModel(data=data)
        top_model.set_current_view_type('json')
        self.assertEqual(nested_model.attached_view, nested_model.views['json'])

    def test_recursive_type_propagation_with_nested_lists(self):
        nested_model = mwdv_generator()
        data = {'a': 1, 'b': [nested_model]}
        top_model = base_model.ReportModel(data=data)
        top_model.set_current_view_type('json')
        self.assertEqual(nested_model.attached_view, nested_model.views['json'])

    def test_recursive_type_propogation_on_recursive_structures(self):
        nested_model = mwdv_generator()
        data = {'a': 1, 'b': [nested_model]}
        nested_model['c'] = data
        top_model = base_model.ReportModel(data=data)
        top_model.set_current_view_type('json')
        self.assertEqual(nested_model.attached_view, nested_model.views['json'])
        del nested_model['c']

    def test_report_of_type(self):
        rep = report.ReportOfType('json')
        rep.add_section(lambda x: str(x), mwdv_generator)
        self.assertEqual('{"int": 1, "string": "value"}', rep.run())

    def test_text_report(self):
        rep = report.TextReport('Test Report')
        rep.add_section('An Important Section', mwdv_generator)
        rep.add_section('Another Important Section', mwdv_generator)
        target_str = '========================================================================\n====                          Test Report                           ====\n========================================================================\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n\n========================================================================\n====                      An Important Section                      ====\n========================================================================\nint = 1\nstring = value\n========================================================================\n====                   Another Important Section                    ====\n========================================================================\nint = 1\nstring = value'
        self.assertEqual(target_str, rep.run())

    def test_to_type(self):
        model = mwdv_generator()
        self.assertEqual('<model><int>1</int><string>value</string></model>', model.to_xml())