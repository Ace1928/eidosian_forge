import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
import requests
import webob
from heat.api.aws import exception
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
def _conf_get(self, name):
    if name in self.conf:
        return self.conf[name]
    else:
        return cfg.CONF.ec2authtoken[name]