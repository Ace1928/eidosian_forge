import datetime
import struct
from six.moves import winreg
from six import text_type
from ._common import tzrangebase
class tzwin(tzwinbase):
    """
    Time zone object created from the zone info in the Windows registry

    These are similar to :py:class:`dateutil.tz.tzrange` objects in that
    the time zone data is provided in the format of a single offset rule
    for either 0 or 2 time zone transitions per year.

    :param: name
        The name of a Windows time zone key, e.g. "Eastern Standard Time".
        The full list of keys can be retrieved with :func:`tzwin.list`.
    """

    def __init__(self, name):
        self._name = name
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            tzkeyname = text_type('{kn}\\{name}').format(kn=TZKEYNAME, name=name)
            with winreg.OpenKey(handle, tzkeyname) as tzkey:
                keydict = valuestodict(tzkey)
        self._std_abbr = keydict['Std']
        self._dst_abbr = keydict['Dlt']
        self._display = keydict['Display']
        tup = struct.unpack('=3l16h', keydict['TZI'])
        stdoffset = -tup[0] - tup[1]
        dstoffset = stdoffset - tup[2]
        self._std_offset = datetime.timedelta(minutes=stdoffset)
        self._dst_offset = datetime.timedelta(minutes=dstoffset)
        self._stdmonth, self._stddayofweek, self._stdweeknumber, self._stdhour, self._stdminute = tup[4:9]
        self._dstmonth, self._dstdayofweek, self._dstweeknumber, self._dsthour, self._dstminute = tup[12:17]
        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = self._get_hasdst()

    def __repr__(self):
        return 'tzwin(%s)' % repr(self._name)

    def __reduce__(self):
        return (self.__class__, (self._name,))