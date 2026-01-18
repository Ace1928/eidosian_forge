import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_submodel_serialization(self):
    sm = mwdv_generator()
    sm.set_current_view_type('text')
    self.model['submodel'] = sm
    target_str = 'int = 1\nstring = value\nsubmodel = \n  int = 1\n  string = value'
    self.assertEqual(target_str, str(self.model))