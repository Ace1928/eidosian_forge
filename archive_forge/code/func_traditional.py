from __future__ import division, unicode_literals
import typing
def traditional(size):
    """Convert a filesize in to a string (powers of 1024, JDEC prefixes).

    In this convention, ``1024 B = 1 KB``.

    This is the format that was used to display the size of DVDs
    (*700 MB* meaning actually about *734 003 200 bytes*) before
    standardisation of IEC units among manufacturers, and still
    used by **Windows** to report the storage capacity of hard
    drives (*279.4 GB* meaning *279.4 × 1024³ bytes*).

    Arguments:
        size (int): A file size.

    Returns:
        `str`: A string containing an abbreviated file size and units.

    Example:
        >>> fs.filesize.traditional(30000)
        '29.3 KB'

    """
    return _to_str(size, ('KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'), 1024)