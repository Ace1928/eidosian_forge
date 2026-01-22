import sys
from contextlib import (
from typing import (
from .utils import (  # namedtuple_with_defaults,
class CommandResult(NamedTuple):
    """Encapsulates the results from a cmd2 app command

    :stdout: str - output captured from stdout while this command is executing
    :stderr: str - output captured from stderr while this command is executing
    :stop: bool - return value of onecmd_plus_hooks after it runs the given
           command line.
    :data: possible data populated by the command.

    Any combination of these fields can be used when developing a scripting API
    for a given command. By default stdout, stderr, and stop will be captured
    for you. If there is additional command specific data, then write that to
    cmd2's last_result member. That becomes the data member of this tuple.

    In some cases, the data member may contain everything needed for a command
    and storing stdout and stderr might just be a duplication of data that
    wastes memory. In that case, the StdSim can be told not to store output
    with its pause_storage member. While this member is True, any output sent
    to StdSim won't be saved in its buffer.

    The code would look like this::

        if isinstance(self.stdout, StdSim):
            self.stdout.pause_storage = True

        if isinstance(sys.stderr, StdSim):
            sys.stderr.pause_storage = True

    See :class:`~cmd2.utils.StdSim` for more information.

    .. note::

       Named tuples are immutable. The contents are there for access,
       not for modification.
    """
    stdout: str = ''
    stderr: str = ''
    stop: bool = False
    data: Any = None

    def __bool__(self) -> bool:
        """Returns True if the command succeeded, otherwise False"""
        if self.data is not None:
            return bool(self.data)
        else:
            return not self.stderr