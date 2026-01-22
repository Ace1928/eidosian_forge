from functools import reduce
from datetime import datetime
import re
def _format_test(string):
    """
    Tests the string format to see whether it fits one of the time datatypes
    @param string: attribute value to test
    @return: a URI for the xsd datatype or the string 'plain'
    """
    for key in _formats:
        for f in _formats[key]:
            try:
                _d = datetime.strptime(string, f)
                return key
            except ValueError:
                pass
    if len(string) > 2 and (string[0] == 'P' or (string[0] == '-' and string[1] == 'P')):
        if string[0] == '-':
            for f in _formats[duration_type]:
                try:
                    _d = datetime.strptime(string, f)
                    return duration_type
                except ValueError:
                    pass
        durs = string.split('T')
        if len(durs) == 2:
            dur = durs[0]
            tm = durs[1]
            td = False
            for f in _formats[duration_type]:
                try:
                    _d = datetime.strptime(dur, f)
                    td = True
                    break
                except ValueError:
                    pass
            if td == True:
                for f in _dur_times:
                    try:
                        _d = datetime.strptime(tm, f)
                        return duration_type
                    except ValueError:
                        pass
            return plain
        else:
            return plain
    try:
        s = string[0:-6]
        tz = string[-5:]
        try:
            _t = datetime.strptime(tz, '%H:%M')
        except ValueError:
            return plain
        for f in _formats[datetime_type]:
            try:
                _d = datetime.strptime(s, f)
                return datetime_type
            except ValueError:
                pass
    except:
        pass
    return plain