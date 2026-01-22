import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _parse_error_from_body(self, response):
    xml_contents = response['body']
    root = self._parse_xml_string_to_dom(xml_contents)
    parsed = self._build_name_to_xml_node(root)
    self._replace_nodes(parsed)
    if root.tag == 'Error':
        metadata = self._populate_response_metadata(response)
        parsed.pop('RequestId', '')
        parsed.pop('HostId', '')
        return {'Error': parsed, 'ResponseMetadata': metadata}
    elif 'RequestId' in parsed:
        parsed['ResponseMetadata'] = {'RequestId': parsed.pop('RequestId')}
    default = {'Error': {'Message': '', 'Code': ''}}
    merge_dicts(default, parsed)
    return default