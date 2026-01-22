import doctest
import collections
def _checkForFourIntOrFloatTuple(arg):
    try:
        if not isinstance(arg[0], (int, float)) or not isinstance(arg[1], (int, float)) or (not isinstance(arg[2], (int, float))) or (not isinstance(arg[3], (int, float))):
            raise PyRectException('argument must be a four-item tuple containing int or float values')
    except:
        raise PyRectException('argument must be a four-item tuple containing int or float values')