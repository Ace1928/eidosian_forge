from libcloud.dns.base import Zone, DNSDriver
from libcloud.dns.types import Provider, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.common.buddyns import BuddyNSResponse, BuddyNSException, BuddyNSConnection
class BuddyNSDNSConnection(BuddyNSConnection):
    responseCls = BuddyNSDNSResponse