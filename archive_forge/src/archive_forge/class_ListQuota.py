import abc
import argparse
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListQuota(neutronV20.NeutronCommand, lister.Lister):
    """List quotas of all tenants who have non-default quota values."""
    resource = 'quota'

    def get_parser(self, prog_name):
        parser = super(ListQuota, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        search_opts = {}
        self.log.debug('search options: %s', search_opts)
        obj_lister = getattr(neutron_client, 'list_%ss' % self.resource)
        data = obj_lister(**search_opts)
        info = []
        collection = self.resource + 's'
        if collection in data:
            info = data[collection]
        _columns = len(info) > 0 and sorted(info[0].keys()) or []
        return (_columns, (utils.get_item_properties(s, _columns) for s in info))