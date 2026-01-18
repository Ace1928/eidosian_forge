import base64
import datetime
from os import remove
from os.path import join
from OpenSSL import crypto
import dateutil.parser
import pytz
import saml2.cryptography.pki
def verify_chain(self, cert_chain_str_list, cert_str):
    """

        :param cert_chain_str_list: Must be a list of certificate strings,
        where the first certificate to be validate
        is in the beginning and the root certificate is last.
        :param cert_str: The certificate to be validated.
        :return:
        """
    for tmp_cert_str in cert_chain_str_list:
        valid, message = self.verify(tmp_cert_str, cert_str)
        if not valid:
            return (False, message)
        else:
            cert_str = tmp_cert_str
        return (True, 'Signed certificate is valid and correctly signed by CA certificate.')