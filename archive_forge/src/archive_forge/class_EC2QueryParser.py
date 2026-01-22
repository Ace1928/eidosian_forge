import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class EC2QueryParser(QueryParser):

    def _inject_response_metadata(self, node, inject_into):
        mapping = self._build_name_to_xml_node(node)
        child_node = mapping.get('requestId')
        if child_node is not None:
            inject_into['ResponseMetadata'] = {'RequestId': child_node.text}

    def _do_error_parse(self, response, shape):
        original = super()._do_error_parse(response, shape)
        if 'RequestID' in original:
            original['ResponseMetadata'] = {'RequestId': original.pop('RequestID')}
        return original

    def _get_error_root(self, original_root):
        for child in original_root:
            if self._node_tag(child) == 'Errors':
                for errors_child in child:
                    if self._node_tag(errors_child) == 'Error':
                        return errors_child
        return original_root