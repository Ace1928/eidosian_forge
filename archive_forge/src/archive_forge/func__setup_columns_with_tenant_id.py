import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def _setup_columns_with_tenant_id(self, display_columns, avail_columns):
    _columns = [x for x in display_columns if x in avail_columns]
    if 'tenant_id' in display_columns:
        return _columns
    if 'tenant_id' not in avail_columns:
        return _columns
    if not self.is_admin_role():
        return _columns
    try:
        pos_id = _columns.index('id')
    except ValueError:
        pos_id = 0
    try:
        pos_name = _columns.index('name')
    except ValueError:
        pos_name = 0
    _columns.insert(max(pos_id, pos_name) + 1, 'tenant_id')
    return _columns