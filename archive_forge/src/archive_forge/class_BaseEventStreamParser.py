import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class BaseEventStreamParser(ResponseParser):

    def _do_parse(self, response, shape):
        final_parsed = {}
        if shape.serialization.get('eventstream'):
            event_type = response['headers'].get(':event-type')
            event_shape = shape.members.get(event_type)
            if event_shape:
                final_parsed[event_type] = self._do_parse(response, event_shape)
        else:
            self._parse_non_payload_attrs(response, shape, shape.members, final_parsed)
            self._parse_payload(response, shape, shape.members, final_parsed)
        return final_parsed

    def _do_error_parse(self, response, shape):
        exception_type = response['headers'].get(':exception-type')
        exception_shape = shape.members.get(exception_type)
        if exception_shape is not None:
            original_parsed = self._initial_body_parse(response['body'])
            body = self._parse_shape(exception_shape, original_parsed)
            error = {'Error': {'Code': exception_type, 'Message': body.get('Message', body.get('message', ''))}}
        else:
            error = {'Error': {'Code': response['headers'].get(':error-code', ''), 'Message': response['headers'].get(':error-message', '')}}
        return error

    def _parse_payload(self, response, shape, member_shapes, final_parsed):
        if shape.serialization.get('event'):
            for name in member_shapes:
                member_shape = member_shapes[name]
                if member_shape.serialization.get('eventpayload'):
                    body = response['body']
                    if member_shape.type_name == 'blob':
                        parsed_body = body
                    elif member_shape.type_name == 'string':
                        parsed_body = body.decode(self.DEFAULT_ENCODING)
                    else:
                        raw_parse = self._initial_body_parse(body)
                        parsed_body = self._parse_shape(member_shape, raw_parse)
                    final_parsed[name] = parsed_body
                    return
            original_parsed = self._initial_body_parse(response['body'])
            body_parsed = self._parse_shape(shape, original_parsed)
            final_parsed.update(body_parsed)

    def _parse_non_payload_attrs(self, response, shape, member_shapes, final_parsed):
        headers = response['headers']
        for name in member_shapes:
            member_shape = member_shapes[name]
            if member_shape.serialization.get('eventheader'):
                if name in headers:
                    value = headers[name]
                    if member_shape.type_name == 'timestamp':
                        value = self._timestamp_parser(value / 1000.0)
                    final_parsed[name] = value

    def _initial_body_parse(self, body_contents):
        raise NotImplementedError('_initial_body_parse')