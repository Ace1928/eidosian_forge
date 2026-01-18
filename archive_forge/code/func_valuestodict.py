import logging
from datetime import datetime
from tzlocal import utils
from tzlocal.windows_tz import win_tz
def valuestodict(key):
    """Convert a registry key's values to a dictionary."""
    result = {}
    size = winreg.QueryInfoKey(key)[1]
    for i in range(size):
        data = winreg.EnumValue(key, i)
        result[data[0]] = data[1]
    return result