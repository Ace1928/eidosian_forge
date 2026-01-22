import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def _stub_get_attr(self, chain, refids, attrs):
    chain.get_output = mock.Mock(side_effect=exception.NotFound)

    def make_fake_res(idx):
        fr = mock.Mock()
        fr.stack = chain.stack
        fr.FnGetRefId.return_value = refids[idx]
        fr.FnGetAtt.return_value = attrs[idx]
        return fr
    fake_res = {str(i): make_fake_res(i) for i in refids}
    chain.nested = mock.Mock(return_value=fake_res)