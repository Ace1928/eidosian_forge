import base64
import datetime
from os import remove
from os.path import join
from OpenSSL import crypto
import dateutil.parser
import pytz
import saml2.cryptography.pki
def write_str_to_file(self, file, str_data):
    with open(file, 'w') as f:
        f.write(str_data)