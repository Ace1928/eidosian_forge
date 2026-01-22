import abc
import errno
import os
import signal
import sys
import time
import eventlet
from eventlet.green import socket
from eventlet.green import ssl
import eventlet.greenio
import eventlet.wsgi
import functools
from oslo_concurrency import processutils
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from paste.deploy import loadwsgi
from routes import middleware
import webob.dec
import webob.exc
from heat.api.aws import exception as aws_exception
from heat.common import exception
from heat.common.i18n import _
from heat.common import serializers
class AppFactory(BasePasteFactory):
    """A Generic paste.deploy app factory.

    This requires heat.app_factory to be set to a callable which returns a
    WSGI app when invoked. The format of the name is <module>:<callable> e.g.

      [app:apiv1app]
      paste.app_factory = heat.common.wsgi:app_factory
      heat.app_factory = heat.api.cfn.v1:API

    The WSGI app constructor must accept a ConfigOpts object and a local config
    dict as its two arguments.
    """
    KEY = 'heat.app_factory'

    def __call__(self, global_conf, **local_conf):
        """The actual paste.app_factory protocol method."""
        factory = self._import_factory(local_conf)
        return factory(self.conf, **local_conf)