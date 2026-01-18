import re
from requests import cookies, utils
from . import _digest_auth_compat as auth
Handle HTTP 407 only once, otherwise give up

        :param r: current response
        :returns: responses, along with the new response
        