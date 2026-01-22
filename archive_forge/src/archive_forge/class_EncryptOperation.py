from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
class EncryptOperation(CryptoOperation):
    """Encrypts a file."""
    keyname = 'public'
    description = 'Encrypts a file. The file must be shorter than the key length in order to be encrypted.'
    operation = 'encrypt'
    operation_past = 'encrypted'
    operation_progressive = 'encrypting'

    def perform_operation(self, indata, pub_key, cli_args=None):
        """Encrypts files."""
        return rsa.encrypt(indata, pub_key)