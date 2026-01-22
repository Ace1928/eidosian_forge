import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class ExistentDomain(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = 'Domain already exists on our servers'
        super().__init__(value, http_code, 408, driver)