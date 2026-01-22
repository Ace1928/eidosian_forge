from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import utils
class HypervisorStatsManager(base.Manager):
    resource_class = HypervisorStats

    @api_versions.wraps('2.0', '2.87')
    def statistics(self):
        """
        Get hypervisor statistics over all compute nodes.
        """
        return self._get('/os-hypervisors/statistics', 'hypervisor_statistics')

    @api_versions.wraps('2.88')
    def statistics(self):
        raise exceptions.UnsupportedVersion(_("The 'statistics' API is removed in API version 2.88 or later."))