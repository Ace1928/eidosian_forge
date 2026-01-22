import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.drivers.openstack import OpenStack_1_1_Response, OpenStack_1_1_Connection
class RackspaceDNSDriver(DNSDriver, OpenStackDriverMixin):
    name = 'Rackspace DNS'
    website = 'http://www.rackspace.com/'
    type = Provider.RACKSPACE
    connectionCls = RackspaceDNSConnection

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='us', **kwargs):
        valid_regions = self.list_regions()
        if region not in valid_regions:
            raise ValueError('Invalid region: %s' % region)
        OpenStackDriverMixin.__init__(self, **kwargs)
        super().__init__(key=key, secret=secret, host=host, port=port, region=region)
    RECORD_TYPE_MAP = {RecordType.A: 'A', RecordType.AAAA: 'AAAA', RecordType.CNAME: 'CNAME', RecordType.MX: 'MX', RecordType.NS: 'NS', RecordType.PTR: 'PTR', RecordType.SRV: 'SRV', RecordType.TXT: 'TXT'}

    @classmethod
    def list_regions(cls):
        return ['us', 'uk']

    def iterate_zones(self):
        offset = 0
        limit = 100
        while True:
            params = {'limit': limit, 'offset': offset}
            response = self.connection.request(action='/domains', params=params).object
            zones_list = response['domains']
            for item in zones_list:
                yield self._to_zone(item)
            if _rackspace_result_has_more(response, len(zones_list), limit):
                offset += limit
            else:
                break

    def iterate_records(self, zone):
        self.connection.set_context({'resource': 'zone', 'id': zone.id})
        offset = 0
        limit = 100
        while True:
            params = {'showRecord': True, 'limit': limit, 'offset': offset}
            response = self.connection.request(action='/domains/%s' % zone.id, params=params).object
            records_list = response['recordsList']
            records = records_list['records']
            for item in records:
                record = self._to_record(data=item, zone=zone)
                yield record
            if _rackspace_result_has_more(records_list, len(records), limit):
                offset += limit
            else:
                break

    def get_zone(self, zone_id):
        self.connection.set_context({'resource': 'zone', 'id': zone_id})
        response = self.connection.request(action='/domains/%s' % zone_id)
        zone = self._to_zone(data=response.object)
        return zone

    def get_record(self, zone_id, record_id):
        zone = self.get_zone(zone_id=zone_id)
        self.connection.set_context({'resource': 'record', 'id': record_id})
        response = self.connection.request(action='/domains/{}/records/{}'.format(zone_id, record_id)).object
        record = self._to_record(data=response, zone=zone)
        return record

    def create_zone(self, domain, type='master', ttl=None, extra=None):
        extra = extra if extra else {}
        if 'email' not in extra:
            raise ValueError('"email" key must be present in extra dictionary')
        payload = {'name': domain, 'emailAddress': extra['email'], 'recordsList': {'records': []}}
        if ttl:
            payload['ttl'] = ttl
        if 'comment' in extra:
            payload['comment'] = extra['comment']
        data = {'domains': [payload]}
        response = self.connection.async_request(action='/domains', method='POST', data=data)
        zone = self._to_zone(data=response.object['response']['domains'][0])
        return zone

    def update_zone(self, zone, domain=None, type=None, ttl=None, extra=None):
        extra = extra if extra else {}
        if domain:
            raise LibcloudError('Domain cannot be changed', driver=self)
        data = {}
        if ttl:
            data['ttl'] = int(ttl)
        if 'email' in extra:
            data['emailAddress'] = extra['email']
        if 'comment' in extra:
            data['comment'] = extra['comment']
        type = type if type else zone.type
        ttl = ttl if ttl else zone.ttl
        self.connection.set_context({'resource': 'zone', 'id': zone.id})
        self.connection.async_request(action='/domains/%s' % zone.id, method='PUT', data=data)
        merged = merge_valid_keys(params=copy.deepcopy(zone.extra), valid_keys=VALID_ZONE_EXTRA_PARAMS, extra=extra)
        updated_zone = get_new_obj(obj=zone, klass=Zone, attributes={'type': type, 'ttl': ttl, 'extra': merged})
        return updated_zone

    def create_record(self, name, zone, type, data, extra=None):
        extra = extra if extra else {}
        name = self._to_full_record_name(domain=zone.domain, name=name)
        data = {'name': name, 'type': self.RECORD_TYPE_MAP[type], 'data': data}
        if 'ttl' in extra:
            data['ttl'] = int(extra['ttl'])
        if 'priority' in extra:
            data['priority'] = int(extra['priority'])
        payload = {'records': [data]}
        self.connection.set_context({'resource': 'zone', 'id': zone.id})
        response = self.connection.async_request(action='/domains/%s/records' % zone.id, data=payload, method='POST').object
        record = self._to_record(data=response['response']['records'][0], zone=zone)
        return record

    def update_record(self, record, name=None, type=None, data=None, extra=None):
        extra = extra if extra else {}
        name = self._to_full_record_name(domain=record.zone.domain, name=record.name)
        payload = {'name': name}
        if data:
            payload['data'] = data
        if 'ttl' in extra:
            payload['ttl'] = extra['ttl']
        if 'comment' in extra:
            payload['comment'] = extra['comment']
        type = type if type is not None else record.type
        data = data if data else record.data
        self.connection.set_context({'resource': 'record', 'id': record.id})
        self.connection.async_request(action='/domains/{}/records/{}'.format(record.zone.id, record.id), method='PUT', data=payload)
        merged = merge_valid_keys(params=copy.deepcopy(record.extra), valid_keys=VALID_RECORD_EXTRA_PARAMS, extra=extra)
        updated_record = get_new_obj(obj=record, klass=Record, attributes={'type': type, 'data': data, 'driver': self, 'extra': merged})
        return updated_record

    def delete_zone(self, zone):
        self.connection.set_context({'resource': 'zone', 'id': zone.id})
        self.connection.async_request(action='/domains/%s' % zone.id, method='DELETE')
        return True

    def delete_record(self, record):
        self.connection.set_context({'resource': 'record', 'id': record.id})
        self.connection.async_request(action='/domains/{}/records/{}'.format(record.zone.id, record.id), method='DELETE')
        return True

    def ex_iterate_ptr_records(self, device):
        """
        Return a generator to iterate over existing PTR Records.

        The ``device`` should be an instance of one of these:
            :class:`libcloud.compute.base.Node`
            :class:`libcloud.loadbalancer.base.LoadBalancer`

        And it needs to have the following ``extra`` fields set:
            service_name - the service catalog name for the device
            uri - the URI pointing to the GET endpoint for the device

        Those are automatically set for you if you got the device from
        the Rackspace driver for that service.

        For example:
            server = rs_compute.ex_get_node_details(id)
            ptr_iter = rs_dns.ex_list_ptr_records(server)

            loadbalancer = rs_lbs.get_balancer(id)
            ptr_iter = rs_dns.ex_list_ptr_records(loadbalancer)

        Note: the Rackspace DNS API docs indicate that the device 'href' is
        optional, but testing does not bear this out. It throws a
        400 Bad Request error if you do not pass in the 'href' from
        the server or loadbalancer.  So ``device`` is required.

        :param device: the device that owns the IP
        :rtype: ``generator`` of :class:`RackspacePTRRecord`
        """
        _check_ptr_extra_fields(device)
        params = {'href': device.extra['uri']}
        service_name = device.extra['service_name']
        self.connection.set_context({'resource': 'ptr_records'})
        try:
            response = self.connection.request(action='/rdns/%s' % service_name, params=params).object
            records = response['records']
            link = dict(rel=service_name, **params)
            for item in records:
                record = self._to_ptr_record(data=item, link=link)
                yield record
        except BaseHTTPError as exc:
            if exc.code == 404:
                return
            raise

    def ex_get_ptr_record(self, service_name, record_id):
        """
        Get a specific PTR record by id.

        :param service_name: the service catalog name of the linked device(s)
                             i.e. cloudLoadBalancers or cloudServersOpenStack
        :param record_id: the id (i.e. PTR-12345) of the PTR record
        :rtype: instance of :class:`RackspacePTRRecord`
        """
        self.connection.set_context({'resource': 'record', 'id': record_id})
        response = self.connection.request(action='/rdns/{}/{}'.format(service_name, record_id)).object
        item = next(iter(response['recordsList']['records']))
        return self._to_ptr_record(data=item, link=response['link'])

    def ex_create_ptr_record(self, device, ip, domain, extra=None):
        """
        Create a PTR record for a specific IP on a specific device.

        The ``device`` should be an instance of one of these:
            :class:`libcloud.compute.base.Node`
            :class:`libcloud.loadbalancer.base.LoadBalancer`

        And it needs to have the following ``extra`` fields set:
            service_name - the service catalog name for the device
            uri - the URI pointing to the GET endpoint for the device

        Those are automatically set for you if you got the device from
        the Rackspace driver for that service.

        For example:
            server = rs_compute.ex_get_node_details(id)
            rs_dns.create_ptr_record(server, ip, domain)

            loadbalancer = rs_lbs.get_balancer(id)
            rs_dns.create_ptr_record(loadbalancer, ip, domain)

        :param device: the device that owns the IP
        :param ip: the IP for which you want to set reverse DNS
        :param domain: the fqdn you want that IP to represent
        :param extra: a ``dict`` with optional extra values:
            ttl - the time-to-live of the PTR record
        :rtype: instance of :class:`RackspacePTRRecord`
        """
        _check_ptr_extra_fields(device)
        if extra is None:
            extra = {}
        data = {'name': domain, 'type': RecordType.PTR, 'data': ip}
        if 'ttl' in extra:
            data['ttl'] = extra['ttl']
        payload = {'recordsList': {'records': [data]}, 'link': {'content': '', 'href': device.extra['uri'], 'rel': device.extra['service_name']}}
        response = self.connection.async_request(action='/rdns', method='POST', data=payload).object
        item = next(iter(response['response']['records']))
        return self._to_ptr_record(data=item, link=payload['link'])

    def ex_update_ptr_record(self, record, domain=None, extra=None):
        """
        Update a PTR record for a specific IP on a specific device.

        If you need to change the domain or ttl, use this API to
        update the record by deleting the old one and creating a new one.

        :param record: the original :class:`RackspacePTRRecord`
        :param domain: the fqdn you want that IP to represent
        :param extra: a ``dict`` with optional extra values:
            ttl - the time-to-live of the PTR record
        :rtype: instance of :class:`RackspacePTRRecord`
        """
        if domain is not None and domain == record.domain:
            domain = None
        if extra is not None:
            extra = dict(extra)
            for key in extra:
                if key in record.extra and record.extra[key] == extra[key]:
                    del extra[key]
        if domain is None and (not extra):
            return record
        _check_ptr_extra_fields(record)
        ip = record.ip
        self.ex_delete_ptr_record(record)
        return self.ex_create_ptr_record(record, ip, domain, extra=extra)

    def ex_delete_ptr_record(self, record):
        """
        Delete an existing PTR Record

        :param record: the original :class:`RackspacePTRRecord`
        :rtype: ``bool``
        """
        _check_ptr_extra_fields(record)
        self.connection.set_context({'resource': 'record', 'id': record.id})
        self.connection.async_request(action='/rdns/%s' % record.extra['service_name'], method='DELETE', params={'href': record.extra['uri'], 'ip': record.ip})
        return True

    def _to_zone(self, data):
        id = data['id']
        domain = data['name']
        type = 'master'
        ttl = data.get('ttl', 0)
        extra = {}
        if 'emailAddress' in data:
            extra['email'] = data['emailAddress']
        if 'comment' in data:
            extra['comment'] = data['comment']
        zone = Zone(id=str(id), domain=domain, type=type, ttl=int(ttl), driver=self, extra=extra)
        return zone

    def _to_record(self, data, zone):
        id = data['id']
        fqdn = data['name']
        name = self._to_partial_record_name(domain=zone.domain, name=fqdn)
        type = self._string_to_record_type(data['type'])
        record_data = data['data']
        extra = {'fqdn': fqdn}
        for key in VALID_RECORD_EXTRA_PARAMS:
            if key in data:
                extra[key] = data[key]
        record = Record(id=str(id), name=name, type=type, data=record_data, zone=zone, driver=self, ttl=extra.get('ttl', None), extra=extra)
        return record

    def _to_ptr_record(self, data, link):
        id = data['id']
        ip = data['data']
        domain = data['name']
        extra = {'uri': link['href'], 'service_name': link['rel']}
        for key in VALID_RECORD_EXTRA_PARAMS:
            if key in data:
                extra[key] = data[key]
        record = RackspacePTRRecord(id=str(id), ip=ip, domain=domain, driver=self, extra=extra)
        return record

    def _to_full_record_name(self, domain, name):
        """
        Build a FQDN from a domain and record name.

        :param domain: Domain name.
        :type domain: ``str``

        :param name: Record name.
        :type name: ``str``
        """
        if name:
            name = '{}.{}'.format(name, domain)
        else:
            name = domain
        return name

    def _to_partial_record_name(self, domain, name):
        """
        Remove domain portion from the record name.

        :param domain: Domain name.
        :type domain: ``str``

        :param name: Full record name (fqdn).
        :type name: ``str``
        """
        if name == domain:
            return None
        name = name.replace('.%s' % domain, '')
        return name

    def _ex_connection_class_kwargs(self):
        kwargs = self.openstack_connection_kwargs()
        kwargs['region'] = self.region
        return kwargs