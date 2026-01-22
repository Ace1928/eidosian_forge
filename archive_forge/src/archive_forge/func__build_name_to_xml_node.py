import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _build_name_to_xml_node(self, parent_node):
    if isinstance(parent_node, list):
        return self._build_name_to_xml_node(parent_node[0])
    xml_dict = {}
    for item in parent_node:
        key = self._node_tag(item)
        if key in xml_dict:
            if isinstance(xml_dict[key], list):
                xml_dict[key].append(item)
            else:
                xml_dict[key] = [xml_dict[key], item]
        else:
            xml_dict[key] = item
    return xml_dict