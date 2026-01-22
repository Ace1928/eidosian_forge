import base64
import datetime
from os import remove
from os.path import join
from OpenSSL import crypto
import dateutil.parser
import pytz
import saml2.cryptography.pki
class WrongInput(Exception):
    pass