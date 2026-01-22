import datetime
from typing import Any, Dict, List, Type, Union, Iterator, Optional
from libcloud import __version__
from libcloud.dns.types import RecordType
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
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