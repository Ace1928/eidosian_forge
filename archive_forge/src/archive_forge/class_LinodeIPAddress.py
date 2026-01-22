from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeIPAddress:

    def __init__(self, inet, public, version, driver, extra=None):
        self.inet = inet
        self.public = public
        self.version = version
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<IPAddress: address=%s, public=%r, version=%s, driver=%s ...>' % (self.inet, self.public, self.version, self.driver.name)