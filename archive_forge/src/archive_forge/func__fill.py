import sys
import datetime
import os.path
from pytz.exceptions import AmbiguousTimeError
from pytz.exceptions import InvalidTimeError
from pytz.exceptions import NonExistentTimeError
from pytz.exceptions import UnknownTimeZoneError
from pytz.lazy import LazyDict, LazyList, LazySet  # noqa
from pytz.tzinfo import unpickler, BaseTzInfo
from pytz.tzfile import build_tzinfo
def _fill(self):
    data = {}
    zone_tab = open_resource('iso3166.tab')
    try:
        for line in zone_tab.readlines():
            line = line.decode('UTF-8')
            if line.startswith('#'):
                continue
            code, name = line.split(None, 1)
            data[code] = name.strip()
        self.data = data
    finally:
        zone_tab.close()