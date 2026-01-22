from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorProcess(Interface):

    def spawnProcess(processProtocol: 'IProcessProtocol', executable: Union[bytes, str], args: Sequence[Union[bytes, str]], env: Optional[Mapping[AnyStr, AnyStr]]=None, path: Union[None, bytes, str]=None, uid: Optional[int]=None, gid: Optional[int]=None, usePTY: bool=False, childFDs: Optional[Mapping[int, Union[int, str]]]=None) -> 'IProcessTransport':
        """
        Spawn a process, with a process protocol.

        Arguments given to this function that are listed as L{bytes} or
        L{unicode} may be encoded or decoded depending on the platform and the
        argument type given.  On UNIX systems (Linux, FreeBSD, macOS) and
        Python 2 on Windows, L{unicode} arguments will be encoded down to
        L{bytes} using the encoding given by L{sys.getfilesystemencoding}, to be
        used with the "narrow" OS APIs.  On Python 3 on Windows, L{bytes}
        arguments will be decoded up to L{unicode} using the encoding given by
        L{sys.getfilesystemencoding} (C{utf8}) and given to Windows's native "wide" APIs.

        @param processProtocol: An object which will be notified of all events
            related to the created process.

        @param executable: the file name to spawn - the full path should be
            used.

        @param args: the command line arguments to pass to the process; a
            sequence of strings.  The first string should be the executable's
            name.

        @param env: the environment variables to pass to the child process.
            The resulting behavior varies between platforms.  If:

                - C{env} is not set:
                  - On POSIX: pass an empty environment.
                  - On Windows: pass L{os.environ}.
                - C{env} is L{None}:
                  - On POSIX: pass L{os.environ}.
                  - On Windows: pass L{os.environ}.
                - C{env} is a L{dict}:
                  - On POSIX: pass the key/value pairs in C{env} as the
                    complete environment.
                  - On Windows: update L{os.environ} with the key/value
                    pairs in the L{dict} before passing it. As a
                    consequence of U{bug #1640
                    <http://twistedmatrix.com/trac/ticket/1640>}, passing
                    keys with empty values in an effort to unset
                    environment variables I{won't} unset them.

        @param path: the path to run the subprocess in - defaults to the
            current directory.

        @param uid: user ID to run the subprocess as.  (Only available on POSIX
            systems.)

        @param gid: group ID to run the subprocess as.  (Only available on
            POSIX systems.)

        @param usePTY: if true, run this process in a pseudo-terminal.
            optionally a tuple of C{(masterfd, slavefd, ttyname)}, in which
            case use those file descriptors.  (Not available on all systems.)

        @param childFDs: A dictionary mapping file descriptors in the new child
            process to an integer or to the string 'r' or 'w'.

            If the value is an integer, it specifies a file descriptor in the
            parent process which will be mapped to a file descriptor (specified
            by the key) in the child process.  This is useful for things like
            inetd and shell-like file redirection.

            If it is the string 'r', a pipe will be created and attached to the
            child at that file descriptor: the child will be able to write to
            that file descriptor and the parent will receive read notification
            via the L{IProcessProtocol.childDataReceived} callback.  This is
            useful for the child's stdout and stderr.

            If it is the string 'w', similar setup to the previous case will
            occur, with the pipe being readable by the child instead of
            writeable.  The parent process can write to that file descriptor
            using L{IProcessTransport.writeToChild}.  This is useful for the
            child's stdin.

            If childFDs is not passed, the default behaviour is to use a
            mapping that opens the usual stdin/stdout/stderr pipes.

        @see: L{twisted.internet.protocol.ProcessProtocol}

        @return: An object which provides L{IProcessTransport}.

        @raise OSError: Raised with errno C{EAGAIN} or C{ENOMEM} if there are
            insufficient system resources to create a new process.
        """