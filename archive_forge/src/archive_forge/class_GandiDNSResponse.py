from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi import GandiResponse, BaseGandiDriver, GandiConnection
class GandiDNSResponse(GandiResponse):
    exceptions = {581042: ZoneDoesNotExistError}