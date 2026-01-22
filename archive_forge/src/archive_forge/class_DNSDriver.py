import datetime
from typing import Any, Dict, List, Type, Union, Iterator, Optional
from libcloud import __version__
from libcloud.dns.types import RecordType
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
class DNSDriver(BaseDriver):
    """
    A base DNSDriver class to derive from

    This class is always subclassed by a specific driver.
    """
    connectionCls = ConnectionUserAndKey
    name = None
    website = None
    RECORD_TYPE_MAP = {}

    def __init__(self, key, secret=None, secure=True, host=None, port=None, **kwargs):
        """
        :param    key: API key or username to used (required)
        :type     key: ``str``

        :param    secret: Secret password to be used (required)
        :type     secret: ``str``

        :param    secure: Whether to use HTTPS or HTTP. Note: Some providers
                only support HTTPS, and it is on by default.
        :type     secure: ``bool``

        :param    host: Override hostname used for connections.
        :type     host: ``str``

        :param    port: Override port used for connections.
        :type     port: ``int``

        :return: ``None``
        """
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, **kwargs)

    def list_record_types(self):
        """
        Return a list of RecordType objects supported by the provider.

        :return: ``list`` of :class:`RecordType`
        """
        return list(self.RECORD_TYPE_MAP.keys())

    def iterate_zones(self):
        """
        Return a generator to iterate over available zones.

        :rtype: ``generator`` of :class:`Zone`
        """
        raise NotImplementedError('iterate_zones not implemented for this driver')

    def list_zones(self):
        """
        Return a list of zones.

        :return: ``list`` of :class:`Zone`
        """
        return list(self.iterate_zones())

    def iterate_records(self, zone):
        """
        Return a generator to iterate over records for the provided zone.

        :param zone: Zone to list records for.
        :type zone: :class:`Zone`

        :rtype: ``generator`` of :class:`Record`
        """
        raise NotImplementedError('iterate_records not implemented for this driver')

    def list_records(self, zone):
        """
        Return a list of records for the provided zone.

        :param zone: Zone to list records for.
        :type zone: :class:`Zone`

        :return: ``list`` of :class:`Record`
        """
        return list(self.iterate_records(zone))

    def get_zone(self, zone_id):
        """
        Return a Zone instance.

        :param zone_id: ID of the required zone
        :type  zone_id: ``str``

        :rtype: :class:`Zone`
        """
        raise NotImplementedError('get_zone not implemented for this driver')

    def get_record(self, zone_id, record_id):
        """
        Return a Record instance.

        :param zone_id: ID of the required zone
        :type  zone_id: ``str``

        :param record_id: ID of the required record
        :type  record_id: ``str``

        :rtype: :class:`Record`
        """
        raise NotImplementedError('get_record not implemented for this driver')

    def create_zone(self, domain, type='master', ttl=None, extra=None):
        """
        Create a new zone.

        :param domain: Zone domain name (e.g. example.com)
        :type domain: ``str``

        :param type: Zone type (master / slave).
        :type  type: ``str``

        :param ttl: TTL for new records. (optional)
        :type  ttl: ``int``

        :param extra: Extra attributes (driver specific). (optional)
        :type extra: ``dict``

        :rtype: :class:`Zone`
        """
        raise NotImplementedError('create_zone not implemented for this driver')

    def update_zone(self, zone, domain, type='master', ttl=None, extra=None):
        """
        Update an existing zone.

        :param zone: Zone to update.
        :type  zone: :class:`Zone`

        :param domain: Zone domain name (e.g. example.com)
        :type  domain: ``str``

        :param type: Zone type (master / slave).
        :type  type: ``str``

        :param ttl: TTL for new records. (optional)
        :type  ttl: ``int``

        :param extra: Extra attributes (driver specific). (optional)
        :type  extra: ``dict``

        :rtype: :class:`Zone`
        """
        raise NotImplementedError('update_zone not implemented for this driver')

    def create_record(self, name, zone, type, data, extra=None):
        """
        Create a new record.

        :param name: Record name without the domain name (e.g. www).
                     Note: If you want to create a record for a base domain
                     name, you should specify empty string ('') for this
                     argument.
        :type  name: ``str``

        :param zone: Zone where the requested record is created.
        :type  zone: :class:`Zone`

        :param type: DNS record type (A, AAAA, ...).
        :type  type: :class:`RecordType`

        :param data: Data for the record (depends on the record type).
        :type  data: ``str``

        :param extra: Extra attributes (driver specific). (optional)
        :type extra: ``dict``

        :rtype: :class:`Record`
        """
        raise NotImplementedError('create_record not implemented for this driver')

    def update_record(self, record, name, type, data, extra=None):
        """
        Update an existing record.

        :param record: Record to update.
        :type  record: :class:`Record`

        :param name: Record name without the domain name (e.g. www).
                     Note: If you want to create a record for a base domain
                     name, you should specify empty string ('') for this
                     argument.
        :type  name: ``str``

        :param type: DNS record type (A, AAAA, ...).
        :type  type: :class:`RecordType`

        :param data: Data for the record (depends on the record type).
        :type  data: ``str``

        :param extra: (optional) Extra attributes (driver specific).
        :type  extra: ``dict``

        :rtype: :class:`Record`
        """
        raise NotImplementedError('update_record not implemented for this driver')

    def delete_zone(self, zone):
        """
        Delete a zone.

        Note: This will delete all the records belonging to this zone.

        :param zone: Zone to delete.
        :type  zone: :class:`Zone`

        :rtype: ``bool``
        """
        raise NotImplementedError('delete_zone not implemented for this driver')

    def delete_record(self, record):
        """
        Delete a record.

        :param record: Record to delete.
        :type  record: :class:`Record`

        :rtype: ``bool``
        """
        raise NotImplementedError('delete_record not implemented for this driver')

    def export_zone_to_bind_format(self, zone):
        """
        Export Zone object to the BIND compatible format.

        :param zone: Zone to export.
        :type  zone: :class:`Zone`

        :return: Zone data in BIND compatible format.
        :rtype: ``str``
        """
        if zone.type != 'master':
            raise ValueError('You can only generate BIND out for master zones')
        lines = []
        records = zone.list_records()
        records = sorted(records, key=Record._get_numeric_id)
        date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        values = {'version': __version__, 'date': date}
        lines.append('; Generated by Libcloud v%(version)s on %(date)s UTC' % values)
        lines.append('$ORIGIN {domain}.'.format(domain=zone.domain))
        lines.append('$TTL {domain_ttl}\n'.format(domain_ttl=zone.ttl))
        for record in records:
            line = self._get_bind_record_line(record=record)
            lines.append(line)
        output = '\n'.join(lines)
        return output

    def export_zone_to_bind_zone_file(self, zone, file_path):
        """
        Export Zone object to the BIND compatible format and write result to a
        file.

        :param zone: Zone to export.
        :type  zone: :class:`Zone`

        :param file_path: File path where the output will be saved.
        :type  file_path: ``str``
        """
        result = self.export_zone_to_bind_format(zone=zone)
        with open(file_path, 'w') as fp:
            fp.write(result)

    def _get_bind_record_line(self, record):
        """
        Generate BIND record line for the provided record.

        :param record: Record to generate the line for.
        :type  record: :class:`Record`

        :return: Bind compatible record line.
        :rtype: ``str``
        """
        parts = []
        if record.name:
            name = '{name}.{domain}'.format(name=record.name, domain=record.zone.domain)
        else:
            name = record.zone.domain
        name += '.'
        ttl = record.extra['ttl'] if 'ttl' in record.extra else record.zone.ttl
        ttl = str(ttl)
        data = record.data
        if record.type in [RecordType.CNAME, RecordType.DNAME, RecordType.MX, RecordType.PTR, RecordType.SRV]:
            if data[len(data) - 1] != '.':
                data += '.'
        if record.type in [RecordType.TXT, RecordType.SPF] and ' ' in data:
            data = data.replace('"', '\\"')
            data = '"%s"' % data
        if record.type in [RecordType.MX, RecordType.SRV]:
            priority = str(record.extra['priority'])
            parts = [name, ttl, 'IN', str(record.type), priority, data]
        else:
            parts = [name, ttl, 'IN', str(record.type), data]
        line = '\t'.join(parts)
        return line

    def _string_to_record_type(self, string):
        """
        Return a string representation of a DNS record type to a
        libcloud RecordType ENUM.

        :rtype: ``str``
        """
        string = string.upper()
        record_type = getattr(RecordType, string)
        return record_type