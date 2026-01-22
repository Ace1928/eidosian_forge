import typing
from enum import IntEnum
from sys import exit as sysexit, stderr, stdout
from typing import Union
class ExitStatus(IntEnum):
    """
    Standard exit status codes for system programs.

    @cvar EX_OK: Successful termination.
    @cvar EX_USAGE: Command line usage error.
    @cvar EX_DATAERR: Data format error.
    @cvar EX_NOINPUT: Cannot open input.
    @cvar EX_NOUSER: Addressee unknown.
    @cvar EX_NOHOST: Host name unknown.
    @cvar EX_UNAVAILABLE: Service unavailable.
    @cvar EX_SOFTWARE: Internal software error.
    @cvar EX_OSERR: System error (e.g., can't fork).
    @cvar EX_OSFILE: Critical OS file missing.
    @cvar EX_CANTCREAT: Can't create (user) output file.
    @cvar EX_IOERR: Input/output error.
    @cvar EX_TEMPFAIL: Temporary failure; the user is invited to retry.
    @cvar EX_PROTOCOL: Remote error in protocol.
    @cvar EX_NOPERM: Permission denied.
    @cvar EX_CONFIG: Configuration error.
    """
    EX_OK = Status.EX_OK
    EX_USAGE = Status.EX_USAGE
    EX_DATAERR = Status.EX_DATAERR
    EX_NOINPUT = Status.EX_NOINPUT
    EX_NOUSER = Status.EX_NOUSER
    EX_NOHOST = Status.EX_NOHOST
    EX_UNAVAILABLE = Status.EX_UNAVAILABLE
    EX_SOFTWARE = Status.EX_SOFTWARE
    EX_OSERR = Status.EX_OSERR
    EX_OSFILE = Status.EX_OSFILE
    EX_CANTCREAT = Status.EX_CANTCREAT
    EX_IOERR = Status.EX_IOERR
    EX_TEMPFAIL = Status.EX_TEMPFAIL
    EX_PROTOCOL = Status.EX_PROTOCOL
    EX_NOPERM = Status.EX_NOPERM
    EX_CONFIG = Status.EX_CONFIG