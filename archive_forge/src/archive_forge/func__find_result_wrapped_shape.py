import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _find_result_wrapped_shape(self, element_name, xml_root_node):
    mapping = self._build_name_to_xml_node(xml_root_node)
    return mapping[element_name]