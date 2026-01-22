from libcloud.dns.base import Zone, Record, DNSDriver, RecordType
from libcloud.dns.types import (
from libcloud.utils.py3 import urlencode
from libcloud.common.dnspod import DNSPodResponse, DNSPodException, DNSPodConnection

        Create a record.

        :param name: Record name without the domain name (e.g. www).
                     Note: If you want to create a record for a base domain
                     name, you should specify empty string ('') for this
                     argument.
        :type  name: ``str``

        :param zone: Zone which the records will be created for.
        :type zone: :class:`Zone`

        :param type: DNS record type ( 'A', 'AAAA', 'CNAME', 'MX', 'NS',
                     'PTR', 'SOA', 'SRV', 'TXT').
        :type  type: :class:`RecordType`

        :param data: Data for the record (depends on the record type).
        :type  data: ``str``

        :param extra: (optional) Extra attributes ('prio', 'ttl').
        :type  extra: ``dict``

        :rtype: :class:`Record`
        