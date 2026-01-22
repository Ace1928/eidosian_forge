import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _build_service_accounts_gce_list(self, service_accounts=None, default_email='default', default_scope='devstorage.read_only'):
    """
        Helper to create service account list for GCE API.

        :keyword  service_accounts: Specify a list of serviceAccounts when
                                       creating the instance. The format is a
                                       list of dictionaries containing email
                                       and list of scopes, e.g.
                                       [{'email':'default',
                                       'scopes':['compute', ...]}, ...]
                                       Scopes can either be full URLs or short
                                       names. If not provided, use the
                                       'default' service account email and a
                                       scope of 'devstorage.read_only'. Also
                                       accepts the aliases defined in
                                       'gcloud compute'.

        :type     service_accounts: ``list`` of ``dict``, ``None`` or an empty
                                    list. ``None` means use a default service
                                    account and an empty list indicates no
                                    service account.

        :return:  list of dictionaries usable in the GCE API.
        :rtype:   ``list`` of ``dict``
        """
    gce_service_accounts = []
    if service_accounts is None:
        gce_service_accounts = [{'email': default_email, 'scopes': [self.AUTH_URL + default_scope]}]
    elif not isinstance(service_accounts, list):
        raise ValueError('service_accounts field is not a list.')
    else:
        for sa in service_accounts:
            gce_service_accounts.append(self._build_service_account_gce_struct(service_account=sa))
    return gce_service_accounts