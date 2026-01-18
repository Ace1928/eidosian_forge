import sys
import re
import six
from ncclient.transport.errors import SessionCloseError, TransportError, PermissionError
from ncclient.transport.ssh import SSHSession
Underlying `paramiko.Transport <http://www.lag.net/paramiko/docs/paramiko.Transport-class.html>`_ object. This makes it possible to call methods like :meth:`~paramiko.Transport.set_keepalive` on it.