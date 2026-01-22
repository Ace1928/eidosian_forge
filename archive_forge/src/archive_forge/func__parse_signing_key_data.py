import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher
import nacl.signing
from paramiko.message import Message
from paramiko.pkey import PKey, OPENSSH_AUTH_MAGIC, _unpad_openssh
from paramiko.util import b
from paramiko.ssh_exception import SSHException, PasswordRequiredException
def _parse_signing_key_data(self, data, password):
    from paramiko.transport import Transport
    message = Message(data)
    if message.get_bytes(len(OPENSSH_AUTH_MAGIC)) != OPENSSH_AUTH_MAGIC:
        raise SSHException('Invalid key')
    ciphername = message.get_text()
    kdfname = message.get_text()
    kdfoptions = message.get_binary()
    num_keys = message.get_int()
    if kdfname == 'none':
        if kdfoptions or ciphername != 'none':
            raise SSHException('Invalid key')
    elif kdfname == 'bcrypt':
        if not password:
            raise PasswordRequiredException('Private key file is encrypted')
        kdf = Message(kdfoptions)
        bcrypt_salt = kdf.get_binary()
        bcrypt_rounds = kdf.get_int()
    else:
        raise SSHException('Invalid key')
    if ciphername != 'none' and ciphername not in Transport._cipher_info:
        raise SSHException('Invalid key')
    public_keys = []
    for _ in range(num_keys):
        pubkey = Message(message.get_binary())
        if pubkey.get_text() != self.name:
            raise SSHException('Invalid key')
        public_keys.append(pubkey.get_binary())
    private_ciphertext = message.get_binary()
    if ciphername == 'none':
        private_data = private_ciphertext
    else:
        cipher = Transport._cipher_info[ciphername]
        key = bcrypt.kdf(password=b(password), salt=bcrypt_salt, desired_key_bytes=cipher['key-size'] + cipher['block-size'], rounds=bcrypt_rounds, ignore_few_rounds=True)
        decryptor = Cipher(cipher['class'](key[:cipher['key-size']]), cipher['mode'](key[cipher['key-size']:]), backend=default_backend()).decryptor()
        private_data = decryptor.update(private_ciphertext) + decryptor.finalize()
    message = Message(_unpad_openssh(private_data))
    if message.get_int() != message.get_int():
        raise SSHException('Invalid key')
    signing_keys = []
    for i in range(num_keys):
        if message.get_text() != self.name:
            raise SSHException('Invalid key')
        public = message.get_binary()
        key_data = message.get_binary()
        signing_key = nacl.signing.SigningKey(key_data[:32])
        assert signing_key.verify_key.encode() == public == public_keys[i] == key_data[32:]
        signing_keys.append(signing_key)
        message.get_binary()
    if len(signing_keys) != 1:
        raise SSHException('Invalid key')
    return signing_keys[0]