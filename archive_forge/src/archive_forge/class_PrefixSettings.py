import collections
from requests import compat
class PrefixSettings(_PrefixSettings):

    def __new__(cls, request, response):
        request = _coerce_to_bytes(request)
        response = _coerce_to_bytes(response)
        return super(PrefixSettings, cls).__new__(cls, request, response)