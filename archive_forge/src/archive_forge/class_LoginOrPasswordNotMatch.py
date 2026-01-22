import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class LoginOrPasswordNotMatch(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = 'Login ID and/or Password you supplied is not on file or' + ' does not match'
        super().__init__(value, http_code, 403, driver)