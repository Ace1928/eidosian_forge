import logging
import queue
import threading
from oslo_utils import eventletutils
from oslo_utils import importutils
def version_is_compatible(imp_version, version):
    """Determine whether versions are compatible.

    :param imp_version: The version implemented
    :param version: The version requested by an incoming message.
    """
    if imp_version is None:
        return True
    if version is None:
        return False
    version_parts = version.split('.')
    imp_version_parts = imp_version.split('.')
    try:
        rev = version_parts[2]
    except IndexError:
        rev = 0
    try:
        imp_rev = imp_version_parts[2]
    except IndexError:
        imp_rev = 0
    if int(version_parts[0]) != int(imp_version_parts[0]):
        return False
    if int(version_parts[1]) > int(imp_version_parts[1]):
        return False
    if int(version_parts[1]) == int(imp_version_parts[1]) and int(rev) > int(imp_rev):
        return False
    return True