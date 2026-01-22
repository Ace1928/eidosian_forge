import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def allocation_show(self, identifier, fields=None, params=''):
    """Show specified baremetal allocation.

        :param String identifier: Name or UUID of the allocation
        :param List fields: List of fields to show
        :param List params: Additional kwargs
        :return: JSON object of allocation
        """
    opts = self.get_opts(fields)
    output = self.openstack('baremetal allocation show {0} {1} {2}'.format(opts, identifier, params))
    return json.loads(output)