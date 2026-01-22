from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
class HostTechJSONAPI(ZoneRecordAPI, JSONAPIHelper):

    def __init__(self, http_helper, token, api='https://api.ns1.hosttech.eu/api/', debug=False):
        """
        Create a new HostTech API instance with given API token.
        """
        JSONAPIHelper.__init__(self, http_helper, token, api=api, debug=debug)

    def _extract_error_message(self, result):
        if result is None:
            return ''
        if isinstance(result, dict):
            res = ''
            if result.get('message'):
                res = '{0} with message "{1}"'.format(res, result['message'])
            if 'errors' in result:
                if isinstance(result['errors'], dict):
                    for k, v in sorted(result['errors'].items()):
                        if isinstance(v, list):
                            v = '; '.join(v)
                        res = '{0} (field "{1}": {2})'.format(res, k, v)
            if res:
                return res
        return ' with data: {0}'.format(result)

    def _create_headers(self):
        return dict(accept='application/json', authorization='Bearer {token}'.format(token=self._token))

    def _list_pagination(self, url, query=None, block_size=100):
        result = []
        offset = 0
        while True:
            query_ = query.copy() if query else dict()
            query_['limit'] = block_size
            query_['offset'] = offset
            res, info = self._get(url, query_, must_have_content=True, expected=[200])
            result.extend(res['data'])
            if len(res['data']) < block_size:
                return result
            offset += block_size

    def get_zone_with_records_by_id(self, id, prefix=NOT_PROVIDED, record_type=NOT_PROVIDED):
        """
        Given a zone ID, return the zone contents with records if found.

        @param id: The zone ID
        @param prefix: The prefix to filter for, if provided. Since None is a valid value,
                       the special constant NOT_PROVIDED indicates that we are not filtering.
        @param record_type: The record type to filter for, if provided
        @return The zone information with records (DNSZoneWithRecords), or None if not found
        """
        result, info = self._get('user/v1/zones/{0}'.format(id), expected=[200, 404], must_have_content=[200])
        if info['status'] == 404:
            return None
        return _create_zone_with_records_from_json(result['data'], prefix=prefix, record_type=record_type)

    def get_zone_with_records_by_name(self, name, prefix=NOT_PROVIDED, record_type=NOT_PROVIDED):
        """
        Given a zone name, return the zone contents with records if found.

        @param name: The zone name (string)
        @param prefix: The prefix to filter for, if provided. Since None is a valid value,
                       the special constant NOT_PROVIDED indicates that we are not filtering.
        @param record_type: The record type to filter for, if provided
        @return The zone information with records (DNSZoneWithRecords), or None if not found
        """
        result = self._list_pagination('user/v1/zones', query=dict(query=name))
        for zone in result:
            if zone['name'] == name:
                result, info = self._get('user/v1/zones/{0}'.format(zone['id']), expected=[200])
                return _create_zone_with_records_from_json(result['data'], prefix=prefix, record_type=record_type)
        return None

    def get_zone_records(self, zone_id, prefix=NOT_PROVIDED, record_type=NOT_PROVIDED):
        """
        Given a zone ID, return a list of records, optionally filtered by the provided criteria.

        @param zone_id: The zone ID
        @param prefix: The prefix to filter for, if provided. Since None is a valid value,
                       the special constant NOT_PROVIDED indicates that we are not filtering.
        @param record_type: The record type to filter for, if provided
        @return A list of DNSrecord objects, or None if zone was not found
        """
        query = dict()
        if record_type is not NOT_PROVIDED:
            query['type'] = record_type.upper()
        result, info = self._get('user/v1/zones/{0}/records'.format(zone_id), query=query, expected=[200, 404], must_have_content=[200])
        if info['status'] == 404:
            return None
        return filter_records([_create_record_from_json(record) for record in result['data']], prefix=prefix, record_type=record_type)

    def get_zone_by_name(self, name):
        """
        Given a zone name, return the zone contents if found.

        @param name: The zone name (string)
        @return The zone information (DNSZone), or None if not found
        """
        result = self._list_pagination('user/v1/zones', query=dict(query=name))
        for zone in result:
            if zone['name'] == name:
                return self.get_zone_by_id(zone['id'])
        return None

    def get_zone_by_id(self, id):
        """
        Given a zone ID, return the zone contents if found.

        @param id: The zone ID
        @return The zone information (DNSZone), or None if not found
        """
        result, info = self._get('user/v1/zones/{0}'.format(id), expected=[200, 404], must_have_content=[200])
        if info['status'] == 404:
            return None
        return _create_zone_from_json(result['data'])

    def add_record(self, zone_id, record):
        """
        Adds a new record to an existing zone.

        @param zone_id: The zone ID
        @param record: The DNS record (DNSRecord)
        @return The created DNS record (DNSRecord)
        """
        data = _record_to_json(record, include_id=False, include_type=True)
        result, dummy = self._post('user/v1/zones/{0}/records'.format(zone_id), data=data, expected=[201])
        return _create_record_from_json(result['data'])

    def update_record(self, zone_id, record):
        """
        Update a record.

        @param zone_id: The zone ID
        @param record: The DNS record (DNSRecord)
        @return The DNS record (DNSRecord)
        """
        if record.id is None:
            raise DNSAPIError('Need record ID to update record!')
        data = _record_to_json(record, include_id=False, include_type=False)
        result, dummy = self._put('user/v1/zones/{0}/records/{1}'.format(zone_id, record.id), data=data, expected=[200])
        return _create_record_from_json(result['data'])

    def delete_record(self, zone_id, record):
        """
        Delete a record.

        @param zone_id: The zone ID
        @param record: The DNS record (DNSRecord)
        @return True in case of success (boolean)
        """
        if record.id is None:
            raise DNSAPIError('Need record ID to delete record!')
        dummy, info = self._delete('user/v1/zones/{0}/records/{1}'.format(zone_id, record.id), must_have_content=False, expected=[204, 404])
        return info['status'] == 204