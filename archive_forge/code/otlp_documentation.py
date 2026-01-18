import collections
from urllib import parse as parser
from oslo_config import cfg
from oslo_serialization import jsonutils
from osprofiler import _utils as utils
from osprofiler.drivers import base
from osprofiler import exc
Create tags an OpenTracing compatible span.

        :param info: Information from OSProfiler trace.
        :returns tags: A dictionary contains standard tags
                       from OpenTracing sematic conventions,
                       and some other custom tags related to http, db calls.
        