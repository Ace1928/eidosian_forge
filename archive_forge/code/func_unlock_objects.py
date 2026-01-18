import os
from typing import Any, List, Tuple
from jeepney import (
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import DBUS_UNKNOWN_METHOD, DBUS_NO_SUCH_OBJECT, \
from secretstorage.dhcrypto import Session, int_to_bytes
from secretstorage.exceptions import ItemNotFoundException, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def unlock_objects(connection: DBusConnection, paths: List[str]) -> bool:
    """Requests unlocking objects specified in `paths`.
    Returns a boolean representing whether the operation was dismissed.

    .. versionadded:: 2.1.2"""
    service = DBusAddressWrapper(SS_PATH, SERVICE_IFACE, connection)
    unlocked_paths, prompt = service.call('Unlock', 'ao', paths)
    if len(prompt) > 1:
        dismissed, (signature, unlocked) = exec_prompt(connection, prompt)
        assert signature == 'ao'
        return dismissed
    return False