import re
from isodate.isoerror import ISO8601Error
from isodate.tzinfo import UTC, FixedOffset, ZERO
def tz_isoformat(dt, format='%Z'):
    """
    return time zone offset ISO 8601 formatted.
    The various ISO formats can be chosen with the format parameter.

    if tzinfo is None returns ''
    if tzinfo is UTC returns 'Z'
    else the offset is rendered to the given format.
    format:
        %h ... +-HH
        %z ... +-HHMM
        %Z ... +-HH:MM
    """
    tzinfo = dt.tzinfo
    if tzinfo is None or tzinfo.utcoffset(dt) is None:
        return ''
    if tzinfo.utcoffset(dt) == ZERO and tzinfo.dst(dt) == ZERO:
        return 'Z'
    tdelta = tzinfo.utcoffset(dt)
    seconds = tdelta.days * 24 * 60 * 60 + tdelta.seconds
    sign = seconds < 0 and '-' or '+'
    seconds = abs(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 99:
        raise OverflowError('can not handle differences > 99 hours')
    if format == '%Z':
        return '%s%02d:%02d' % (sign, hours, minutes)
    elif format == '%z':
        return '%s%02d%02d' % (sign, hours, minutes)
    elif format == '%h':
        return '%s%02d' % (sign, hours)
    raise ValueError('unknown format string "%s"' % format)