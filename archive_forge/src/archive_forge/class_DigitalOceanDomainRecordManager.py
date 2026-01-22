from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DigitalOceanDomainRecordManager(DigitalOceanHelper, object):

    def __init__(self, module):
        super(DigitalOceanDomainRecordManager, self).__init__(module)
        self.module = module
        self.domain = module.params.get('domain').lower()
        self.records = self.__get_all_records()
        self.payload = self.__build_payload()
        self.force_update = module.params.get('force_update', False)
        self.record_id = module.params.get('record_id', None)

    def check_credentials(self):
        response = self.get('account')
        if response.status_code == 401:
            self.module.fail_json(msg='Failed to login using oauth_token, please verify validity of oauth_token')

    def verify_domain(self):
        response = self.get('domains/%s' % self.domain)
        status_code = response.status_code
        json = response.json
        if status_code not in (200, 404):
            self.module.fail_json(msg='Error getting domain [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})
        elif status_code == 404:
            self.module.fail_json(msg="No domain named '%s' found. Please create a domain first" % self.domain)

    def __get_all_records(self):
        records = []
        page = 1
        while True:
            response = self.get('domains/%(domain)s/records?page=%(page)s' % {'domain': self.domain, 'page': page})
            status_code = response.status_code
            json = response.json
            if status_code != 200:
                self.module.fail_json(msg='Error getting domain records [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})
            for record in json['domain_records']:
                records.append(dict([(str(k), v) for k, v in record.items()]))
            if 'pages' in json['links'] and 'next' in json['links']['pages']:
                page += 1
            else:
                break
        return records

    def __normalize_data(self):
        if self.payload['type'] in ['CNAME', 'MX', 'SRV', 'CAA'] and self.payload['data'] != '@' and (not self.payload['data'].endswith('.')):
            data = '%s.' % self.payload['data']
        else:
            data = self.payload['data']
        return data

    def __find_record_by_id(self, record_id):
        for record in self.records:
            if record['id'] == record_id:
                return record
        return None

    def __get_matching_records(self):
        """Collect exact and similar records

        It returns an exact record if there is any match along with the record_id.
        It also returns multiple records if there is no exact match
        """
        for record in self.records:
            r = dict(record)
            del r['id']
            if r == self.payload:
                return (r, record['id'], None)
        similar_records = []
        for record in self.records:
            if record['type'] == self.payload['type'] and record['name'] == self.payload['name']:
                similar_records.append(record)
        if similar_records:
            return (None, None, similar_records)
        return (None, None, None)

    def __create_record(self):
        self.payload['data'] = self.__normalize_data()
        response = self.post('domains/%s/records' % self.domain, data=self.payload)
        status_code = response.status_code
        json = response.json
        if status_code == 201:
            changed = True
            return (changed, json['domain_record'])
        else:
            self.module.fail_json(msg='Error creating domain record [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})

    def create_or_update_record(self):
        if self.record_id:
            changed, result = self.__update_record(self.record_id)
            return (changed, result)
        record, record_id, similar_records = self.__get_matching_records()
        if not record and (not similar_records):
            changed, result = self.__create_record()
            return (changed, result)
        if not record and similar_records:
            if len(similar_records) == 1:
                if self.force_update:
                    record_id = similar_records[0]['id']
                    changed, result = self.__update_record(record_id)
                else:
                    changed, result = self.__create_record()
                return (changed, result)
            else:
                if self.force_update:
                    self.module.fail_json(msg="Can't update record, too many similar records: %s" % similar_records)
                else:
                    changed, result = self.__create_record()
                return (changed, result)
        else:
            changed = False
            result = 'Record has been already created'
            return (changed, result)

    def __update_record(self, record_id):
        self.payload['data'] = self.__normalize_data()
        record = self.__find_record_by_id(record_id)
        if record:
            response = self.put('domains/%(domain)s/records/%(record_id)s' % {'domain': self.domain, 'record_id': record_id}, data=self.payload)
            status_code = response.status_code
            json = response.json
            if status_code == 200:
                changed = True
                return (changed, json['domain_record'])
            else:
                self.module.fail_json(msg='Error updating domain record [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})
        else:
            self.module.fail_json(msg='Error updating domain record. Record does not exist. [%s]' % record_id)

    def __build_payload(self):
        payload = dict(data=self.module.params.get('data'), flags=self.module.params.get('flags'), name=self.module.params.get('name'), port=self.module.params.get('port'), priority=self.module.params.get('priority'), type=self.module.params.get('type'), tag=self.module.params.get('tag'), ttl=self.module.params.get('ttl'), weight=self.module.params.get('weight'))
        if payload['type'] != 'TXT' and payload['data']:
            payload['data'] = payload['data'].lower()
        if payload['data'] == self.domain:
            payload['data'] = '@'
        return payload

    def delete_record(self):
        if self.record_id:
            record = self.__find_record_by_id(self.record_id)
            record_id = self.record_id
        else:
            record, record_id, similar_records = self.__get_matching_records()
            if not record and similar_records:
                if len(similar_records) == 1:
                    record, record_id = (similar_records[0], similar_records[0]['id'])
                else:
                    self.module.fail_json(msg="Can't delete record, too many similar records: %s" % similar_records)
        if not record:
            changed = False
            return (changed, record)
        else:
            response = self.delete('domains/%(domain)s/records/%(id)s' % {'domain': self.domain, 'id': record_id})
            status_code = response.status_code
            json = response.json
            if status_code == 204:
                changed = True
                msg = 'Successfully deleted %s' % record['name']
                return (changed, msg)
            else:
                self.module.fail_json(msg='Error deleting domain record. [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})