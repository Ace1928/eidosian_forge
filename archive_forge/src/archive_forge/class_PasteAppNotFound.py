import copy
import os
import socket
import eventlet
import eventlet.wsgi
import greenlet
from paste import deploy
import routes.middleware
import webob.dec
import webob.exc
from oslo_log import log as logging
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import service
from oslo_service import sslutils
class PasteAppNotFound(Exception):

    def __init__(self, name, path):
        msg = _("Could not load paste app '%(name)s' from %(path)s") % {'name': name, 'path': path}
        super(PasteAppNotFound, self).__init__(msg)