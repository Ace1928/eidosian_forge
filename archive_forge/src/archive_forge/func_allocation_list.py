import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def allocation_list(self, fields=None, params=''):
    """List baremetal allocations.

        :param List fields: List of fields to show
        :param String params: Additional kwargs
        :return: list of JSON allocation objects
        """
    opts = self.get_opts(fields=fields)
    output = self.openstack('baremetal allocation list {0} {1}'.format(opts, params))
    return json.loads(output)