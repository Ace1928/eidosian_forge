import binascii
import lzma
import platform
import sys
class DefaultFilters(Constant):
    """Default filter values."""
    ARCHIVE_FILTER = [{'id': FILTER_X86}, {'id': FILTER_LZMA2, 'preset': 7 | PRESET_DEFAULT}]
    ENCODED_HEADER_FILTER = [{'id': FILTER_LZMA2, 'preset': 7 | PRESET_DEFAULT}]
    ENCRYPTED_ARCHIVE_FILTER = [{'id': FILTER_LZMA2, 'preset': 7 | PRESET_DEFAULT}, {'id': FILTER_CRYPTO_AES256_SHA256}]
    ENCRYPTED_HEADER_FILTER = [{'id': FILTER_CRYPTO_AES256_SHA256}]