import calendar
import copy
import http.cookiejar as http_cookiejar
from http.cookies import SimpleCookie
import logging
import re
import time
from urllib.parse import urlencode
from urllib.parse import urlparse
import requests
from saml2 import SAMLError
from saml2 import class_name
from saml2.pack import make_soap_enveloped_saml_thingy
from saml2.time_util import utc_now
def _since_epoch(cdate):
    """
    :param cdate: date format 'Wed, 06-Jun-2012 01:34:34 GMT'
    :return: UTC time
    """
    if len(cdate) < 29:
        if len(cdate) < 5:
            return utc_now()
    cdate = cdate[5:]
    t = -1
    for time_format in TIME_FORMAT:
        try:
            t = time.strptime(cdate, time_format)
        except ValueError:
            pass
        else:
            break
    if t == -1:
        err = f'ValueError: Date "{cdate}" does not match any of: {TIME_FORMAT}'
        raise Exception(err)
    return calendar.timegm(t)