import binascii
from .settings import ChangedSetting, _setting_code_from_int
def _bytes_representation(data):
    """
    Converts a bytestring into something that is safe to print on all Python
    platforms.

    This function is relatively expensive, so it should not be called on the
    mainline of the code. It's safe to use in things like object repr methods
    though.
    """
    if data is None:
        return None
    hex = binascii.hexlify(data)
    if not isinstance(hex, str):
        hex = hex.decode('ascii')
    return hex