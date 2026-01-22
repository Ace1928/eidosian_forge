import webob
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import strutils
import requests
def _deny_request(self, code):
    error_table = {'AccessDenied': (401, 'Access denied'), 'InvalidURI': (400, 'Could not parse the specified URI')}
    resp = webob.Response(content_type='text/xml')
    resp.status = error_table[code][0]
    error_msg = '<?xml version="1.0" encoding="UTF-8"?>\r\n<Error>\r\n  <Code>%s</Code>\r\n  <Message>%s</Message>\r\n</Error>\r\n' % (code, error_table[code][1])
    error_msg = error_msg.encode()
    resp.body = error_msg
    return resp