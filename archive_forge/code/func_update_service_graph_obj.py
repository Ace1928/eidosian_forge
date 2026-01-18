from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def update_service_graph_obj(self, service_graph_obj):
    """update filter with more information"""
    service_graph_obj['serviceGraphRef'] = self.dict_from_ref(service_graph_obj.get('serviceGraphRef'))
    for service_node in service_graph_obj['serviceNodesRelationship']:
        service_node.get('consumerConnector')['bdRef'] = self.dict_from_ref(service_node.get('consumerConnector').get('bdRef'))
        service_node.get('providerConnector')['bdRef'] = self.dict_from_ref(service_node.get('providerConnector').get('bdRef'))
        service_node['serviceNodeRef'] = self.dict_from_ref(service_node.get('serviceNodeRef'))
    if service_graph_obj.get('serviceGraphContractRelationRef'):
        del service_graph_obj['serviceGraphContractRelationRef']