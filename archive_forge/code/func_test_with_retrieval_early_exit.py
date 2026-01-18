import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.continue_retrieval')
@mock.patch('oslo_vmware.vim_util.cancel_retrieval')
def test_with_retrieval_early_exit(self, cancel_retrieval, continue_retrieval):
    vim = mock.Mock()
    retrieve_result = mock.Mock()
    with vim_util.WithRetrieval(vim, retrieve_result):
        pass
    cancel_retrieval.assert_called_once_with(vim, retrieve_result)