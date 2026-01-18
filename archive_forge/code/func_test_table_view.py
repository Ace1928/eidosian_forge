import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_table_view(self):
    column_names = ['Column A', 'Column B']
    column_values = ['a', 'b']
    attached_view = text_generic.TableView(column_names, column_values, 'table')
    self.model = base_model.ReportModel(data={}, attached_view=attached_view)
    self.model['table'] = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    target_str = '             Column A              |             Column B               \n------------------------------------------------------------------------\n                 1                 |                 2                  \n                 3                 |                 4                  \n'
    self.assertEqual(target_str, str(self.model))