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
class FilterFactory(AppFactory):
    """A Generic paste.deploy filter factory.

    This requires heat.filter_factory to be set to a callable which returns a
    WSGI filter when invoked. The format is <module>:<callable> e.g.

      [filter:cache]
      paste.filter_factory = heat.common.wsgi:filter_factory
      heat.filter_factory = heat.api.middleware.cache:CacheFilter

    The WSGI filter constructor must accept a WSGI app, a ConfigOpts object and
    a local config dict as its three arguments.
    """
    KEY = 'heat.filter_factory'

    def __call__(self, global_conf, **local_conf):
        """The actual paste.filter_factory protocol method."""
        factory = self._import_factory(local_conf)

        def filter(app):
            return factory(app, self.conf, **local_conf)
        return filter