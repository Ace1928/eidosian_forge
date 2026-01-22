from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import ctypes
from ctypes import windll  # pylint: disable=g-importing-member
from ctypes import wintypes  # pylint: disable=g-importing-member
class WinCrypt(object):
    """Base Class for Windows encryption functions."""

    def __init__(self):
        self.crypt_acquire_context = advapi32.CryptAcquireContextW
        self.crypt_acquire_context.restype = wintypes.BOOL
        self.crypt_acquire_context.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self.crypt_release_context = advapi32.CryptReleaseContext
        self.crypt_release_context.restype = wintypes.BOOL
        self.crypt_release_context.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.crypt_gen_key = advapi32.CryptGenKey
        self.crypt_gen_key.restype = wintypes.BOOL
        self.crypt_gen_key.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        self.crypt_destroy_key = advapi32.CryptDestroyKey
        self.crypt_destroy_key.restype = wintypes.BOOL
        self.crypt_destroy_key.argtypes = [wintypes.HANDLE]
        self.crypt_decrypt = advapi32.CryptDecrypt
        self.crypt_decrypt.restype = wintypes.BOOL
        self.crypt_decrypt.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD, ctypes.POINTER(wintypes.BYTE), ctypes.POINTER(wintypes.DWORD)]
        self.crypt_export_key = advapi32.CryptExportKey
        self.crypt_export_key.restype = wintypes.BOOL
        self.crypt_export_key.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.BYTE), ctypes.POINTER(wintypes.DWORD)]

    def GetKeyPair(self):
        """Returns a handle for an RSA key pair."""
        crypt_provider_handle = wintypes.HANDLE()
        key_container_name = None
        provider = MS_ENHANCED_PROV
        provider_type = PROV_RSA_FULL
        acquire_context_flags = CRYPT_VERIFYCONTEXT
        algorithm_id = CALG_RSA_KEYX
        key_handle = wintypes.HANDLE()
        if not self.crypt_acquire_context(ctypes.byref(crypt_provider_handle), key_container_name, provider, provider_type, acquire_context_flags):
            raise WindowsException
        gen_key_flags = RSA_KEY_LENGTH << 16
        if not self.crypt_gen_key(crypt_provider_handle, algorithm_id, gen_key_flags, key_handle):
            raise WindowsException()
        return key_handle

    def GetPublicKey(self, key):
        """Returns the public key for the referenced private key handle."""
        user_crypto_key = None
        key_type = PUBLICKEYBLOB
        export_key_flags = 0
        key_data = None
        key_len = ctypes.c_ulong()
        self.crypt_export_key(key, user_crypto_key, key_type, export_key_flags, key_data, ctypes.byref(key_len))
        byte_array_type = ctypes.c_byte * key_len.value
        key_data = byte_array_type()
        if not self.crypt_export_key(key, user_crypto_key, key_type, export_key_flags, key_data, ctypes.byref(key_len)):
            raise WindowsException()
        public_key = (ctypes.c_char * key_len.value).from_buffer(key_data)
        return public_key

    def DecryptMessage(self, key, enc_message, destroy_key=True):
        """Returns a decrypted message from the given encrypted message and key.

    Can optionally destroy the key (used only on Windows).

    Args:
      key: An openssl key pair (private key) or a Windows key handle.
      enc_message: A base64 encoded encrypted message.
      destroy_key: If True, the key pointed to by the key handle is destroyed.

    Returns:
      Decrypted version of enc_message

    Raises:
      WindowsException: If message fails to decrypt
    """
        decoded_message = base64.b64decode(enc_message)
        little_endian_message = decoded_message[::-1]
        data_len = ctypes.c_ulong(len(little_endian_message))
        data_buf = (ctypes.c_byte * data_len.value).from_buffer_copy(little_endian_message)
        hash_object = None
        final = True
        decrypt_flags = CRYPT_OAEP
        if not self.crypt_decrypt(key, hash_object, final, decrypt_flags, data_buf, ctypes.byref(data_len)):
            raise WindowsException(data_len)
        message = (ctypes.c_char * data_len.value).from_buffer(data_buf)
        if destroy_key:
            self.crypt_destroy_key(key)
        return message.value

    def GetModulusExponentFromPublicKey(self, public_key):
        """Returns a base64 encoded modulus and exponent from the public key."""
        modulus = public_key[20:][::-1]
        exponent = public_key[16:20][::-1]
        b64_mod = base64.b64encode(modulus)
        b64_exp = base64.b64encode(exponent)
        return (b64_mod, b64_exp)