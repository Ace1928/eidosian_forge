import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
class TestJinjaView(base.BaseTestCase):
    TEMPL_STR = 'int is {{ int }}, string is {{ string }}'
    MM_OPEN, MM_FILE = get_open_mocks(TEMPL_STR)

    def setUp(self):
        super(TestJinjaView, self).setUp()
        self.model = base_model.ReportModel(data={'int': 1, 'string': 'value'})

    @mock.mock_open(MM_OPEN)
    def test_load_from_file(self):
        self.model.attached_view = jv.JinjaView(path='a/b/c/d.jinja.txt')
        self.assertEqual('int is 1, string is value', str(self.model))
        self.MM_FILE.assert_called_with_once('a/b/c/d.jinja.txt')

    def test_direct_pass(self):
        self.model.attached_view = jv.JinjaView(text=self.TEMPL_STR)
        self.assertEqual('int is 1, string is value', str(self.model))

    def test_load_from_class(self):

        class TmpJinjaView(jv.JinjaView):
            VIEW_TEXT = TestJinjaView.TEMPL_STR
        self.model.attached_view = TmpJinjaView()
        self.assertEqual('int is 1, string is value', str(self.model))

    def test_is_deepcopiable(self):
        view_orig = jv.JinjaView(text=self.TEMPL_STR)
        view_cpy = copy.deepcopy(view_orig)
        self.assertIsNot(view_orig, view_cpy)