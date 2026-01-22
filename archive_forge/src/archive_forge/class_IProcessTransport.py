from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IProcessTransport(ITransport):
    """
    A process transport.
    """
    pid = Attribute('From before L{IProcessProtocol.makeConnection} is called to before L{IProcessProtocol.processEnded} is called, C{pid} is an L{int} giving the platform process ID of this process.  C{pid} is L{None} at all other times.')

    def closeStdin() -> None:
        """
        Close stdin after all data has been written out.
        """

    def closeStdout() -> None:
        """
        Close stdout.
        """

    def closeStderr() -> None:
        """
        Close stderr.
        """

    def closeChildFD(descriptor: int) -> None:
        """
        Close a file descriptor which is connected to the child process, identified
        by its FD in the child process.
        """

    def writeToChild(childFD: int, data: bytes) -> None:
        """
        Similar to L{ITransport.write} but also allows the file descriptor in
        the child process which will receive the bytes to be specified.

        @param childFD: The file descriptor to which to write.
        @param data: The bytes to write.

        @raise KeyError: If C{childFD} is not a file descriptor that was mapped
            in the child when L{IReactorProcess.spawnProcess} was used to create
            it.
        """

    def loseConnection() -> None:
        """
        Close stdin, stderr and stdout.
        """

    def signalProcess(signalID: Union[str, int]) -> None:
        """
        Send a signal to the process.

        @param signalID: can be
          - one of C{"KILL"}, C{"TERM"}, or C{"INT"}.
              These will be implemented in a
              cross-platform manner, and so should be used
              if possible.
          - an integer, where it represents a POSIX
              signal ID.

        @raise twisted.internet.error.ProcessExitedAlready: If the process has
            already exited.
        @raise OSError: If the C{os.kill} call fails with an errno different
            from C{ESRCH}.
        """