import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def _save_pkcs1_pem(self):
    """Saves a PKCS#1 PEM-encoded private key file.

        :return: contents of a PEM-encoded file that contains the private key.
        :rtype: bytes
        """
    der = self._save_pkcs1_der()
    return rsa.pem.save_pem(der, b'RSA PRIVATE KEY')