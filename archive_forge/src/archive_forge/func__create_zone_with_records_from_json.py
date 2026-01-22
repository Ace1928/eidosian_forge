from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _create_zone_with_records_from_json(source, prefix=NOT_PROVIDED, record_type=NOT_PROVIDED):
    return DNSZoneWithRecords(_create_zone_from_json(source), filter_records([_create_record_from_json(record) for record in source['records']], prefix=prefix, record_type=record_type))