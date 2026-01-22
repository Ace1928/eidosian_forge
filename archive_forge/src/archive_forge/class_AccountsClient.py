from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.billing import utils
class AccountsClient(object):
    """High-level client for billing accounts service."""

    def __init__(self, client=None, messages=None):
        self.client = client or utils.GetClient()
        self.messages = messages or self.client.MESSAGES_MODULE
        self._service = self.client.billingAccounts

    def Get(self, account_ref):
        return self._service.Get(self.messages.CloudbillingBillingAccountsGetRequest(name=account_ref.RelativeName()))

    def List(self, limit=None):
        return list_pager.YieldFromList(self._service, self.messages.CloudbillingBillingAccountsListRequest(), field='billingAccounts', batch_size_attribute='pageSize', limit=limit)