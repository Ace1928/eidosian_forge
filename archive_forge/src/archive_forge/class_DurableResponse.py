import re
from typing import Dict, List
from xml.etree import ElementTree as ET  # noqa
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
class DurableResponse(XmlResponse):
    errors = []
    objects = []

    def __init__(self, response, connection):
        super().__init__(response=response, connection=connection)
        self.objects, self.errors = self.parse_body_and_error()
        if self.errors:
            raise self._make_excp(self.errors[0])

    def parse_body_and_error(self):
        """
        Used to parse body from httplib.HttpResponse object.
        """
        objects = []
        errors = []
        error_dict = {}
        extra = {}
        zone_dict = {}
        record_dict = {}
        xml_obj = self.parse_body()
        envelop_body = list(xml_obj)[0]
        method_resp = list(envelop_body)[0]
        if 'Fault' in method_resp.tag:
            fault = [fault for fault in list(method_resp) if fault.tag == 'faultstring'][0]
            error_dict['ERRORMESSAGE'] = fault.text.strip()
            error_dict['ERRORCODE'] = self.status
            errors.append(error_dict)
        if 'listZonesResponse' in method_resp.tag:
            answer = list(method_resp)[0]
            for element in answer:
                zone_dict['id'] = list(element)[0].text
                objects.append(zone_dict)
                zone_dict = {}
        if 'listRecordsResponse' in method_resp.tag:
            answer = list(method_resp)[0]
            for element in answer:
                for child in list(element):
                    if child.tag == 'id':
                        record_dict['id'] = child.text.strip()
                objects.append(record_dict)
                record_dict = {}
        if 'getZoneResponse' in method_resp.tag:
            for child in list(method_resp):
                if child.tag == 'origin':
                    zone_dict['id'] = child.text.strip()
                    zone_dict['domain'] = child.text.strip()
                elif child.tag == 'ttl':
                    zone_dict['ttl'] = int(child.text.strip())
                elif child.tag == 'retry':
                    extra['retry'] = int(child.text.strip())
                elif child.tag == 'expire':
                    extra['expire'] = int(child.text.strip())
                elif child.tag == 'minimum':
                    extra['minimum'] = int(child.text.strip())
                else:
                    if child.text:
                        extra[child.tag] = child.text.strip()
                    else:
                        extra[child.tag] = ''
                    zone_dict['extra'] = extra
            objects.append(zone_dict)
        if 'getRecordResponse' in method_resp.tag:
            answer = list(method_resp)[0]
            for child in list(method_resp):
                if child.tag == 'id' and child.text:
                    record_dict['id'] = child.text.strip()
                elif child.tag == 'name' and child.text:
                    record_dict['name'] = child.text.strip()
                elif child.tag == 'type' and child.text:
                    record_dict['type'] = child.text.strip()
                elif child.tag == 'data' and child.text:
                    record_dict['data'] = child.text.strip()
                elif child.tag == 'aux' and child.text:
                    record_dict['aux'] = child.text.strip()
                elif child.tag == 'ttl' and child.text:
                    record_dict['ttl'] = child.text.strip()
            if not record_dict:
                error_dict['ERRORMESSAGE'] = 'Record does not exist'
                error_dict['ERRORCODE'] = 404
                errors.append(error_dict)
            objects.append(record_dict)
            record_dict = {}
        if 'createZoneResponse' in method_resp.tag:
            answer = list(method_resp)[0]
            if answer.tag == 'return' and answer.text:
                record_dict['id'] = answer.text.strip()
            objects.append(record_dict)
        if 'deleteRecordResponse' in method_resp.tag:
            answer = list(method_resp)[0]
            if 'Record does not exists' in answer.text.strip():
                errors.append({'ERRORMESSAGE': answer.text.strip(), 'ERRORCODE': self.status})
        if 'createRecordResponse' in method_resp.tag:
            answer = list(method_resp)[0]
            record_dict['id'] = answer.text.strip()
            objects.append(record_dict)
            record_dict = {}
        return (objects, errors)

    def parse_body(self):
        self._fix_response()
        body = super().parse_body()
        return body

    def success(self):
        """
        Used to determine if the request was successful.
        """
        return len(self.errors) == 0

    def _make_excp(self, error):
        return DurableDNSException(error['ERRORCODE'], error['ERRORMESSAGE'])

    def _fix_response(self):
        items = re.findall('<ns1:.+ xmlns:ns1="">', self.body, flags=0)
        for item in items:
            parts = item.split(' ')
            prefix = parts[0].replace('<', '').split(':')[1]
            new_item = '<' + prefix + '>'
            close_tag = '</' + parts[0].replace('<', '') + '>'
            new_close_tag = '</' + prefix + '>'
            self.body = self.body.replace(item, new_item)
            self.body = self.body.replace(close_tag, new_close_tag)