import json
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.common.exceptions import BaseHTTPError

        Update an existing record.

        :param record: Record to update.
        :type  record: :class:`Record`

        :param name: FQDN of the new record, for example "www.example.com".
        :type  name: ``str``

        :param type: DNS record type (A, AAAA, ...).
        :type  type: :class:`RecordType`

        :param data: Data for the record (depends on the record type).
        :type  data: ``str``

        :param extra: (optional) Extra attributes (driver specific).
        :type  extra: ``dict``

        :rtype: :class:`Record`
        