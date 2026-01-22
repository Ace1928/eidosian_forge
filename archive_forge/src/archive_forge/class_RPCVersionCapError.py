import abc
import logging
from oslo_config import cfg
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _metrics as metrics
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
class RPCVersionCapError(exceptions.MessagingException):

    def __init__(self, version, version_cap):
        self.version = version
        self.version_cap = version_cap
        msg = 'Requested message version, %(version)s is incompatible.  It needs to be equal in major version and less than or equal in minor version as the specified version cap %(version_cap)s.' % dict(version=self.version, version_cap=self.version_cap)
        super(RPCVersionCapError, self).__init__(msg)