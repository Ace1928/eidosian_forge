from os import getpid
from typing import Dict, List, Mapping, Optional, Sequence
from attrs import Factory, define
def _parseDescriptors(start: int, environ: Mapping[str, str]) -> List[int]:
    """
    Parse the I{LISTEN_FDS} environment variable supplied by systemd.

    @param start: systemd provides only a count of the number of descriptors
        that have been inherited.  This is the integer value of the first
        inherited descriptor.  Subsequent inherited descriptors are numbered
        counting up from here.  See L{ListenFDs._START}.

    @param environ: The environment variable mapping in which to look for the
        value to parse.

    @return: The integer values of the inherited file descriptors, in order.
    """
    try:
        count = int(environ['LISTEN_FDS'])
    except (KeyError, ValueError):
        return []
    else:
        descriptors = list(range(start, start + count))
        del environ['LISTEN_PID'], environ['LISTEN_FDS']
        return descriptors