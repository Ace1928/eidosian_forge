import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import requests
import webob.dec
from keystonemiddleware.i18n import _
def _ec2_error_response(self, code, message):
    """Helper to construct an EC2 compatible error message."""
    self._logger.debug('EC2 error response: %(code)s: %(message)s', {'code': code, 'message': message})
    resp = webob.Response()
    resp.status = 400
    resp.headers['Content-Type'] = 'text/xml'
    error_msg = str('<?xml version="1.0"?>\n<Response><Errors><Error><Code>%s</Code><Message>%s</Message></Error></Errors></Response>' % (code, message))
    error_msg = error_msg.encode()
    resp.body = error_msg
    return resp