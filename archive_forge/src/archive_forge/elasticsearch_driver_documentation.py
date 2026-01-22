from urllib import parse as parser
from oslo_config import cfg
from osprofiler.drivers import base
from osprofiler import exc
Retrieves and parses notification from Elasticsearch.

        :param base_id: Base id of trace elements.
        