import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def _build_verify_list(self, fake_res_assoc, param=None):
    verifylist = [('resource_association_id', fake_res_assoc['id']), ('bgpvpn', self.fake_bgpvpn['id'])]
    if param is not None:
        verifylist.append(param)
    return verifylist