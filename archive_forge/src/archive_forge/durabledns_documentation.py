from xml.etree.ElementTree import tostring
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib, ensure_string
from libcloud.common.durabledns import (
from libcloud.common.durabledns import _schema_builder as api_schema_builder

        Delete a record.

        :param record: Record to delete.
        :type  record: :class:`Record`

        :rtype: ``bool``
        