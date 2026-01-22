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
def _get_unix_socket(self, socket_file, socket_mode, backlog):
    sock = eventlet.listen(socket_file, family=socket.AF_UNIX, backlog=backlog)
    if socket_mode is not None:
        os.chmod(socket_file, socket_mode)
    LOG.info('%(name)s listening on %(socket_file)s:', {'name': self.name, 'socket_file': socket_file})
    return sock