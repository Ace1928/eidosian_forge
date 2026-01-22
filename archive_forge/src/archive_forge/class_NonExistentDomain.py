import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class NonExistentDomain(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = 'Domain name supplied is not in your account'
        super().__init__(value, http_code, 405, driver)