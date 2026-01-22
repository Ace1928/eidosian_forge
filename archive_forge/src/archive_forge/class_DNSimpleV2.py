from __future__ import absolute_import, division, print_function
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
class DNSimpleV2:
    """class which uses dnsimple-python >= 2"""

    def __init__(self, account_email, account_api_token, sandbox, module):
        """init"""
        self.module = module
        self.account_email = account_email
        self.account_api_token = account_api_token
        self.sandbox = sandbox
        self.pagination_per_page = 30
        self.dnsimple_client()
        self.dnsimple_account()

    def dnsimple_client(self):
        """creates a dnsimple client object"""
        if self.account_email and self.account_api_token:
            client = Client(sandbox=self.sandbox, email=self.account_email, access_token=self.account_api_token, user_agent='ansible/community.general')
        else:
            msg = 'Option account_email or account_api_token not provided. Dnsimple authentication with a .dnsimple config file is not supported with dnsimple-python>=2.0.0'
            raise DNSimpleException(msg)
        client.identity.whoami()
        self.client = client

    def dnsimple_account(self):
        """select a dnsimple account. If a user token is used for authentication,
        this user must only have access to a single account"""
        account = self.client.identity.whoami().data.account
        if not account:
            accounts = Accounts(self.client).list_accounts().data
            if len(accounts) != 1:
                msg = 'The provided dnsimple token is a user token with multiple accounts.Use an account token or a user token with access to a single account.See https://support.dnsimple.com/articles/api-access-token/'
                raise DNSimpleException(msg)
            account = accounts[0]
        self.account = account

    def get_all_domains(self):
        """returns a list of all domains"""
        domain_list = self._get_paginated_result(self.client.domains.list_domains, account_id=self.account.id)
        return [d.__dict__ for d in domain_list]

    def get_domain(self, domain):
        """returns a single domain by name or id"""
        try:
            dr = self.client.domains.get_domain(self.account.id, domain).data.__dict__
        except DNSimpleException as e:
            exception_string = str(e.message)
            if re.match('^Domain .+ not found$', exception_string):
                dr = None
            else:
                raise
        return dr

    def create_domain(self, domain):
        """create a single domain"""
        return self.client.domains.create_domain(self.account.id, domain).data.__dict__

    def delete_domain(self, domain):
        """delete a single domain"""
        self.client.domains.delete_domain(self.account.id, domain)

    def get_records(self, zone, dnsimple_filter=None):
        """return dns resource records which match a specified filter"""
        records_list = self._get_paginated_result(self.client.zones.list_records, account_id=self.account.id, zone=zone, filter=dnsimple_filter)
        return [d.__dict__ for d in records_list]

    def delete_record(self, domain, rid):
        """delete a single dns resource record"""
        self.client.zones.delete_record(self.account.id, domain, rid)

    def update_record(self, domain, rid, ttl=None, priority=None):
        """update a single dns resource record"""
        zr = ZoneRecordUpdateInput(ttl=ttl, priority=priority)
        result = self.client.zones.update_record(self.account.id, str(domain), str(rid), zr).data.__dict__
        return result

    def create_record(self, domain, name, record_type, content, ttl=None, priority=None):
        """create a single dns resource record"""
        zr = ZoneRecordInput(name=name, type=record_type, content=content, ttl=ttl, priority=priority)
        return self.client.zones.create_record(self.account.id, str(domain), zr).data.__dict__

    def _get_paginated_result(self, operation, **options):
        """return all results of a paginated api response"""
        records_pagination = operation(per_page=self.pagination_per_page, **options).pagination
        result_list = []
        for page in range(1, records_pagination.total_pages + 1):
            page_data = operation(per_page=self.pagination_per_page, page=page, **options).data
            result_list.extend(page_data)
        return result_list