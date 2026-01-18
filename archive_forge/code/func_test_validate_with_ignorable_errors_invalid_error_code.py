from unittest import mock
from oslo_messaging.rpc import dispatcher
import webob
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine.clients.os import glance
from heat.engine import environment
from heat.engine.hot import template as hot_tmpl
from heat.engine import resources
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_validate_with_ignorable_errors_invalid_error_code(self):
    engine = service.EngineService('a', 't')
    invalide_error_code = '123456'
    invalid_codes = ['99001', invalide_error_code]
    res = engine.validate_template(self.ctx, mock.MagicMock(), {}, ignorable_errors=invalid_codes)
    msg = _('Invalid codes in ignore_errors : %s') % [invalide_error_code]
    ex = webob.exc.HTTPBadRequest(explanation=msg)
    self.assertIsInstance(res, webob.exc.HTTPBadRequest)
    self.assertEqual(ex.explanation, res.explanation)