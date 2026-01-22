import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.common.google import GoogleResponse, GoogleBaseConnection, ResourceNotFoundError
def _cleanup_domain(self, domain):
    domain = re.sub('[^a-zA-Z0-9-]', '-', domain)
    if domain[-1] == '-':
        domain = domain[:-1]
    return domain