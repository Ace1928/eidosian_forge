from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
class OpensshKeypair(object):
    """Container for OpenSSH encoded asymmetric key pairs"""

    @classmethod
    def generate(cls, keytype='rsa', size=None, passphrase=None, comment=None):
        """Returns an Openssh_Keypair object generated using the supplied parameters or defaults to a RSA-2048 key

           :keytype: One of rsa, dsa, ecdsa, ed25519
           :size: The key length for newly generated keys
           :passphrase: Secret of type Bytes used to encrypt the newly generated private key
           :comment: Comment for a newly generated OpenSSH public key
        """
        if comment is None:
            comment = '%s@%s' % (getuser(), gethostname())
        asym_keypair = AsymmetricKeypair.generate(keytype, size, passphrase)
        openssh_privatekey = cls.encode_openssh_privatekey(asym_keypair, 'SSH')
        openssh_publickey = cls.encode_openssh_publickey(asym_keypair, comment)
        fingerprint = calculate_fingerprint(openssh_publickey)
        return cls(asym_keypair=asym_keypair, openssh_privatekey=openssh_privatekey, openssh_publickey=openssh_publickey, fingerprint=fingerprint, comment=comment)

    @classmethod
    def load(cls, path, passphrase=None, no_public_key=False):
        """Returns an Openssh_Keypair object loaded from the supplied file path

           :path: A path to an existing private key to be loaded
           :passphrase: Secret used to decrypt the private key being loaded
           :no_public_key: Set 'True' to only load a private key and automatically populate the matching public key
        """
        if no_public_key:
            comment = ''
        else:
            comment = extract_comment(path + '.pub')
        asym_keypair = AsymmetricKeypair.load(path, passphrase, 'SSH', 'SSH', no_public_key)
        openssh_privatekey = cls.encode_openssh_privatekey(asym_keypair, 'SSH')
        openssh_publickey = cls.encode_openssh_publickey(asym_keypair, comment)
        fingerprint = calculate_fingerprint(openssh_publickey)
        return cls(asym_keypair=asym_keypair, openssh_privatekey=openssh_privatekey, openssh_publickey=openssh_publickey, fingerprint=fingerprint, comment=comment)

    @staticmethod
    def encode_openssh_privatekey(asym_keypair, key_format):
        """Returns an OpenSSH encoded private key for a given keypair

           :asym_keypair: Asymmetric_Keypair from the private key is extracted
           :key_format: Format of the encoded private key.
        """
        if key_format == 'SSH':
            if not HAS_OPENSSH_PRIVATE_FORMAT:
                privatekey_format = serialization.PrivateFormat.PKCS8
            else:
                privatekey_format = serialization.PrivateFormat.OpenSSH
        elif key_format == 'PKCS8':
            privatekey_format = serialization.PrivateFormat.PKCS8
        elif key_format == 'PKCS1':
            if asym_keypair.key_type == 'ed25519':
                raise InvalidKeyFormatError('ed25519 keys cannot be represented in PKCS1 format')
            privatekey_format = serialization.PrivateFormat.TraditionalOpenSSL
        else:
            raise InvalidKeyFormatError('The accepted private key formats are SSH, PKCS8, and PKCS1')
        encoded_privatekey = asym_keypair.private_key.private_bytes(encoding=serialization.Encoding.PEM, format=privatekey_format, encryption_algorithm=asym_keypair.encryption_algorithm)
        return encoded_privatekey

    @staticmethod
    def encode_openssh_publickey(asym_keypair, comment):
        """Returns an OpenSSH encoded public key for a given keypair

           :asym_keypair: Asymmetric_Keypair from the public key is extracted
           :comment: Comment to apply to the end of the returned OpenSSH encoded public key
        """
        encoded_publickey = asym_keypair.public_key.public_bytes(encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH)
        validate_comment(comment)
        encoded_publickey += (' %s' % comment).encode(encoding=_TEXT_ENCODING) if comment else b''
        return encoded_publickey

    def __init__(self, asym_keypair, openssh_privatekey, openssh_publickey, fingerprint, comment):
        """
           :asym_keypair: An Asymmetric_Keypair object from which the OpenSSH encoded keypair is derived
           :openssh_privatekey: An OpenSSH encoded private key
           :openssh_privatekey: An OpenSSH encoded public key
           :fingerprint: The fingerprint of the OpenSSH encoded public key of this keypair
           :comment: Comment applied to the OpenSSH public key of this keypair
        """
        self.__asym_keypair = asym_keypair
        self.__openssh_privatekey = openssh_privatekey
        self.__openssh_publickey = openssh_publickey
        self.__fingerprint = fingerprint
        self.__comment = comment

    def __eq__(self, other):
        if not isinstance(other, OpensshKeypair):
            return NotImplemented
        return self.asymmetric_keypair == other.asymmetric_keypair and self.comment == other.comment

    @property
    def asymmetric_keypair(self):
        """Returns the underlying asymmetric key pair of this OpenSSH encoded key pair"""
        return self.__asym_keypair

    @property
    def private_key(self):
        """Returns the OpenSSH formatted private key of this key pair"""
        return self.__openssh_privatekey

    @property
    def public_key(self):
        """Returns the OpenSSH formatted public key of this key pair"""
        return self.__openssh_publickey

    @property
    def size(self):
        """Returns the size of the private key of this key pair"""
        return self.__asym_keypair.size

    @property
    def key_type(self):
        """Returns the key type of this key pair"""
        return self.__asym_keypair.key_type

    @property
    def fingerprint(self):
        """Returns the fingerprint (SHA256 Hash) of the public key of this key pair"""
        return self.__fingerprint

    @property
    def comment(self):
        """Returns the comment applied to the OpenSSH formatted public key of this key pair"""
        return self.__comment

    @comment.setter
    def comment(self, comment):
        """Updates the comment applied to the OpenSSH formatted public key of this key pair

           :comment: Text to update the OpenSSH public key comment
        """
        validate_comment(comment)
        self.__comment = comment
        encoded_comment = (' %s' % self.__comment).encode(encoding=_TEXT_ENCODING) if self.__comment else b''
        self.__openssh_publickey = b' '.join(self.__openssh_publickey.split(b' ', 2)[:2]) + encoded_comment
        return self.__openssh_publickey

    def update_passphrase(self, passphrase):
        """Updates the passphrase used to encrypt the private key of this keypair

           :passphrase: Text secret used for encryption
        """
        self.__asym_keypair.update_passphrase(passphrase)
        self.__openssh_privatekey = OpensshKeypair.encode_openssh_privatekey(self.__asym_keypair, 'SSH')