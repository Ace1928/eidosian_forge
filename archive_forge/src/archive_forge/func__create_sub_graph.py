import json
import os
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
from osprofiler.cmd import cliutils
from osprofiler.drivers import base
from osprofiler import exc
def _create_sub_graph(root):
    rid = _create_node(root['info'])
    for child in root['children']:
        cid = _create_sub_graph(child)
        dot.edge(rid, cid)
    return rid