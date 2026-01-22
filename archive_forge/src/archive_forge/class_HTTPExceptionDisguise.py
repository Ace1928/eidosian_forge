import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class HTTPExceptionDisguise(Exception):
    """Disguises HTTP exceptions.

    They can be handled by the webob fault application in the wsgi pipeline.
    """
    safe = True

    def __init__(self, exception):
        self.exc = exception
        self.tb = sys.exc_info()[2]