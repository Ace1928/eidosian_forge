import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi_live import (

    Create a new record with multiple values.

    :param data: Record values (depends on the record type)
    :type  data: ``list`` (of ``str``)

    :return: ``list`` of :class:`Record`s
    