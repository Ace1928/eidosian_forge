from castellan.common import exception
from castellan.common import utils
from castellan.tests import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
def test_oslo_context_to_keystone_token(self):
    auth_token_value = '16bd612f28ec479b8ffe8e124fc37b43'
    project_id_value = '00c6ef5ad2984af2acd7d42c299935c0'
    ctxt = context.RequestContext(auth_token=auth_token_value, project_id=project_id_value)
    ks_token_context = utils.credential_factory(context=ctxt)
    ks_token_context_class = ks_token_context.__class__.__name__
    self.assertEqual('KeystoneToken', ks_token_context_class)
    self.assertEqual(auth_token_value, ks_token_context.token)
    self.assertEqual(project_id_value, ks_token_context.project_id)