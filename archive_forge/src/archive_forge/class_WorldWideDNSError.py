import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.types import LibcloudError
from libcloud.common.worldwidedns import WorldWideDNSConnection
class WorldWideDNSError(LibcloudError):

    def __repr__(self):
        return '<WorldWideDNSError in ' + repr(self.driver) + ' ' + repr(self.value) + '>'