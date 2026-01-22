import collections
import datetime
import time
from urllib import parse as parser
from oslo_config import cfg
from oslo_serialization import jsonutils
from osprofiler import _utils as utils
from osprofiler.drivers import base
from osprofiler import exc
def _get_service_name(self, conf, project, service):
    prefix = conf.profiler_jaeger.service_name_prefix
    if prefix:
        return '{}-{}-{}'.format(prefix, project, service)
    return '{}-{}'.format(project, service)