import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.common.google import GoogleResponse, GoogleBaseConnection, ResourceNotFoundError
class GoogleDNSResponse(GoogleResponse):
    pass