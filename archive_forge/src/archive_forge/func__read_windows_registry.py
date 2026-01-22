import os
import sys
import posixpath
import urllib.parse
@classmethod
def _read_windows_registry(cls, add_type):

    def enum_types(mimedb):
        i = 0
        while True:
            try:
                ctype = _winreg.EnumKey(mimedb, i)
            except OSError:
                break
            else:
                if '\x00' not in ctype:
                    yield ctype
            i += 1
    with _winreg.OpenKey(_winreg.HKEY_CLASSES_ROOT, '') as hkcr:
        for subkeyname in enum_types(hkcr):
            try:
                with _winreg.OpenKey(hkcr, subkeyname) as subkey:
                    if not subkeyname.startswith('.'):
                        continue
                    mimetype, datatype = _winreg.QueryValueEx(subkey, 'Content Type')
                    if datatype != _winreg.REG_SZ:
                        continue
                    add_type(mimetype, subkeyname)
            except OSError:
                continue