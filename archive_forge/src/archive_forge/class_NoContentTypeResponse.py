from inspect import getfullargspec
from oslo_config import cfg
import webob.dec
import webob.request
import webob.response
class NoContentTypeResponse(webob.response.Response):
    default_content_type = None