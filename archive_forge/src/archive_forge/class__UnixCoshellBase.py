from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
class _UnixCoshellBase(six.with_metaclass(abc.ABCMeta, _CoshellBase)):
    """The unix local coshell base class.

  Attributes:
    _shell: The coshell subprocess object.
  """
    SHELL_STATUS_EXIT = 'x'
    SHELL_STATUS_FD = 9
    SHELL_STDIN_FD = 8

    def __init__(self):
        super(_UnixCoshellBase, self).__init__()
        self.status = None
        self._status_fd = None
        self._shell = None

    @staticmethod
    def _Quote(command):
        """Quotes command in single quotes so it can be eval'd in coshell."""
        return "'{}'".format(command.replace("'", "'\\''"))

    def _Exited(self):
        """Raises the coshell exit exception."""
        try:
            self._WriteLine(':')
        except (IOError, OSError, ValueError):
            pass
        status = self._ShellStatus(self._shell.returncode)
        raise CoshellExitError('The coshell exited [status={}].'.format(status), status=status)

    def _ReadLine(self):
        """Reads and returns a decoded stripped line from the coshell."""
        return self._Decode(self._shell.stdout.readline()).strip()

    def _ReadStatusChar(self):
        """Reads and returns one encoded character from the coshell status fd."""
        return os.read(self._status_fd, 1)

    def _WriteLine(self, line):
        """Writes an encoded line to the coshell."""
        self._shell.communicate(self._Encode(line + '\n'))

    def _SendCommand(self, command):
        """Sends command to the coshell for execution."""
        try:
            self._shell.stdin.write(self._Encode(command + '\n'))
            self._shell.stdin.flush()
        except (IOError, OSError, ValueError):
            self._Exited()

    def _GetStatus(self):
        """Gets the status of the last command sent to the coshell."""
        line = []
        shell_status_exit = self.SHELL_STATUS_EXIT.encode('ascii')
        while True:
            c = self._ReadStatusChar()
            if c in (None, b'\n', shell_status_exit):
                break
            line.append(c)
        status_string = self._Decode(b''.join(line))
        if not status_string.isdigit() or c == shell_status_exit:
            self._Exited()
        return int(status_string)

    def _GetModes(self):
        """Syncs the user settable modes of interest to the Coshell.

    Calls self._set_modes_callback if it was specified and any mode changed.
    """
        changed = False
        if self.Run('set -o | grep -q "^vi.*on"', check_modes=False) == 0:
            if self._edit_mode != 'vi':
                changed = True
                self._edit_mode = 'vi'
        elif self._edit_mode != 'emacs':
            changed = True
            self._edit_mode = 'emacs'
        ignore_eof = self._ignore_eof
        self._ignore_eof = self.Run('set -o | grep -q "^ignoreeof.*on"', check_modes=False) == 0
        if self._ignore_eof != ignore_eof:
            changed = True
        if changed and self._set_modes_callback:
            self._set_modes_callback()

    def GetPwd(self):
        """Gets the coshell pwd, sets local pwd, returns the pwd, None on error."""
        pwd = self.Communicate(['printf "$PWD\\n\\n"'], quote=False)
        if len(pwd) == 1:
            try:
                os.chdir(pwd[0])
                return pwd[0]
            except OSError:
                pass
        return None

    def _GetUserConfigDefaults(self):
        """Consults the user shell config for defaults."""
        self._SendCommand('COSHELL_VERSION={coshell_version};_status() {{ return $1; }};[[ -f $HOME/.bashrc ]] && source $HOME/.bashrc;trap \'echo $?{exit} >&{fdstatus}\' 0;trap ":" 2;{get_completions_init}'.format(coshell_version=COSHELL_VERSION, exit=self.SHELL_STATUS_EXIT, fdstatus=self.SHELL_STATUS_FD, get_completions_init=_GET_COMPLETIONS_INIT))
        self._SendCommand('set -o monitor 2>/dev/null')
        self._SendCommand('shopt -s expand_aliases 2>/dev/null')
        self._GetModes()
        self._SendCommand('true')

    @abc.abstractmethod
    def _Run(self, command, check_modes=True):
        """Runs command in the coshell and waits for it to complete."""
        pass

    def Run(self, command, check_modes=True):
        """Runs command in the coshell and waits for it to complete."""
        status = 130
        sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            status = self._Run(command, check_modes=check_modes)
        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGINT, sigint)
        return status

    def GetCompletions(self, args, prefix=False):
        """Returns the list of completion choices for args.

    Args:
      args: The list of command line argument strings to complete.
      prefix: Complete the last arg as a command prefix.

    Returns:
      The list of completions for args.
    """
        if prefix:
            completions = self.Communicate(['__coshell_get_executable_completions__', args[-1]])
        else:
            completions = self.Communicate(['__coshell_get_completions__'] + args)
        return sorted(set(completions))

    def Interrupt(self):
        """Sends the interrupt signal to the coshell."""
        self._shell.send_signal(signal.SIGINT)