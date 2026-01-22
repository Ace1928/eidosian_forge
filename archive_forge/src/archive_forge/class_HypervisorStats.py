from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import utils
class HypervisorStats(base.Resource):

    def __repr__(self):
        return '<HypervisorStats: %d Hypervisor%s>' % (self.count, 's' if self.count != 1 else '')