import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
class VMwareSudsTest(base.TestCase):

    def setUp(self):
        super(VMwareSudsTest, self).setUp()

        def new_client_init(self, url, **kwargs):
            return
        mock.patch.object(suds.client.Client, '__init__', new=new_client_init).start()
        self.addCleanup(mock.patch.stopall)
        self.vim = self._vim_create()

    def _mock_getattr(self, attr_name):

        class fake_service_content(object):

            def __init__(self):
                self.ServiceContent = {}
                self.ServiceContent.fake = 'fake'
        self.assertEqual('RetrieveServiceContent', attr_name)
        return lambda obj, **kwargs: fake_service_content()

    def _vim_create(self):
        with mock.patch.object(vim.Vim, '__getattr__', self._mock_getattr):
            return vim.Vim()

    def test_exception_with_deepcopy(self):
        self.assertIsNotNone(self.vim)
        self.assertRaises(exceptions.VimAttributeException, copy.deepcopy, self.vim)