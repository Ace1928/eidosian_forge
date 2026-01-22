from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
def _set_all_params_cidr(self, args={}):
    name = args.get('name') or 'my-name'
    description = args.get('description') or 'my-desc'
    endpoint_type = args.get('type') or 'cidr'
    endpoints = args.get('endpoints') or ['10.0.0.0/24', '20.0.0.0/24']
    tenant_id = args.get('project_id') or 'my-tenant'
    arglist = ['--description', description, '--type', endpoint_type, '--value', '10.0.0.0/24', '--value', '20.0.0.0/24', '--project', tenant_id, name]
    verifylist = [('description', description), ('type', endpoint_type), ('endpoints', endpoints), ('project', tenant_id), ('name', name)]
    return (arglist, verifylist)