import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class NewUserNotValid(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = 'New userid is not valid'
        super().__init__(value, http_code, 414, driver)