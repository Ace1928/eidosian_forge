import _imp
import _io
import sys
import _warnings
import marshal
@classmethod
def _search_registry(cls, fullname):
    if cls.DEBUG_BUILD:
        registry_key = cls.REGISTRY_KEY_DEBUG
    else:
        registry_key = cls.REGISTRY_KEY
    key = registry_key.format(fullname=fullname, sys_version='%d.%d' % sys.version_info[:2])
    try:
        with cls._open_registry(key) as hkey:
            filepath = winreg.QueryValue(hkey, '')
    except OSError:
        return None
    return filepath