import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_recursive_type_propagation_with_nested_models(self):
    model = mwdv_generator()
    model['submodel'] = mwdv_generator()
    model.set_current_view_type('json')
    self.assertEqual(model.submodel.views['json'], model.submodel.attached_view)