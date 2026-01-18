import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_text_report(self):
    rep = report.TextReport('Test Report')
    rep.add_section('An Important Section', mwdv_generator)
    rep.add_section('Another Important Section', mwdv_generator)
    target_str = '========================================================================\n====                          Test Report                           ====\n========================================================================\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n\n========================================================================\n====                      An Important Section                      ====\n========================================================================\nint = 1\nstring = value\n========================================================================\n====                   Another Important Section                    ====\n========================================================================\nint = 1\nstring = value'
    self.assertEqual(target_str, rep.run())