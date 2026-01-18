import logging
import re
Given a datetime, returns a datetime tuple

    Parameters
    ----------
    text: string to be parsed

    Returns
    -------
        (int, int , int, int, int, int, int, int):
            datetime tuple: (year, month, day, hour, minute, second, microsecond, utcoffset in minutes or None)
    