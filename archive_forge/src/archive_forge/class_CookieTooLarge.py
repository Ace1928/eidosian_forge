import hmac, base64, random, time, warnings
from functools import reduce
from paste.request import get_cookies
class CookieTooLarge(RuntimeError):

    def __init__(self, content, cookie):
        RuntimeError.__init__('Signed cookie exceeds maximum size of 4096')
        self.content = content
        self.cookie = cookie