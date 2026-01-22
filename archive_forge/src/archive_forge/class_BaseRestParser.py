import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class BaseRestParser(ResponseParser):

    def _do_parse(self, response, shape):
        final_parsed = {}
        final_parsed['ResponseMetadata'] = self._populate_response_metadata(response)
        self._add_modeled_parse(response, shape, final_parsed)
        return final_parsed

    def _add_modeled_parse(self, response, shape, final_parsed):
        if shape is None:
            return final_parsed
        member_shapes = shape.members
        self._parse_non_payload_attrs(response, shape, member_shapes, final_parsed)
        self._parse_payload(response, shape, member_shapes, final_parsed)

    def _do_modeled_error_parse(self, response, shape):
        final_parsed = {}
        self._add_modeled_parse(response, shape, final_parsed)
        return final_parsed

    def _populate_response_metadata(self, response):
        metadata = {}
        headers = response['headers']
        if 'x-amzn-requestid' in headers:
            metadata['RequestId'] = headers['x-amzn-requestid']
        elif 'x-amz-request-id' in headers:
            metadata['RequestId'] = headers['x-amz-request-id']
            metadata['HostId'] = headers.get('x-amz-id-2', '')
        return metadata

    def _parse_payload(self, response, shape, member_shapes, final_parsed):
        if 'payload' in shape.serialization:
            payload_member_name = shape.serialization['payload']
            body_shape = member_shapes[payload_member_name]
            if body_shape.serialization.get('eventstream'):
                body = self._create_event_stream(response, body_shape)
                final_parsed[payload_member_name] = body
            elif body_shape.type_name in ['string', 'blob']:
                body = response['body']
                if isinstance(body, bytes):
                    body = body.decode(self.DEFAULT_ENCODING)
                final_parsed[payload_member_name] = body
            else:
                original_parsed = self._initial_body_parse(response['body'])
                final_parsed[payload_member_name] = self._parse_shape(body_shape, original_parsed)
        else:
            original_parsed = self._initial_body_parse(response['body'])
            body_parsed = self._parse_shape(shape, original_parsed)
            final_parsed.update(body_parsed)

    def _parse_non_payload_attrs(self, response, shape, member_shapes, final_parsed):
        headers = response['headers']
        for name in member_shapes:
            member_shape = member_shapes[name]
            location = member_shape.serialization.get('location')
            if location is None:
                continue
            elif location == 'statusCode':
                final_parsed[name] = self._parse_shape(member_shape, response['status_code'])
            elif location == 'headers':
                final_parsed[name] = self._parse_header_map(member_shape, headers)
            elif location == 'header':
                header_name = member_shape.serialization.get('name', name)
                if header_name in headers:
                    final_parsed[name] = self._parse_shape(member_shape, headers[header_name])

    def _parse_header_map(self, shape, headers):
        parsed = {}
        prefix = shape.serialization.get('name', '').lower()
        for header_name in headers:
            if header_name.lower().startswith(prefix):
                name = header_name[len(prefix):]
                parsed[name] = headers[header_name]
        return parsed

    def _initial_body_parse(self, body_contents):
        raise NotImplementedError('_initial_body_parse')

    def _handle_string(self, shape, value):
        parsed = value
        if is_json_value_header(shape):
            decoded = base64.b64decode(value).decode(self.DEFAULT_ENCODING)
            parsed = json.loads(decoded)
        return parsed

    def _handle_list(self, shape, node):
        location = shape.serialization.get('location')
        if location == 'header' and (not isinstance(node, list)):
            node = [e.strip() for e in node.split(',')]
        return super()._handle_list(shape, node)