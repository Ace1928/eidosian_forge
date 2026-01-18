from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
def update_passphrase(self, passphrase):
    """Updates the passphrase used to encrypt the private key of this keypair

           :passphrase: Text secret used for encryption
        """
    self.__asym_keypair.update_passphrase(passphrase)
    self.__openssh_privatekey = OpensshKeypair.encode_openssh_privatekey(self.__asym_keypair, 'SSH')