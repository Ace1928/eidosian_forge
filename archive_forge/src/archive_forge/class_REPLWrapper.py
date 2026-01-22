import os.path
import signal
import sys
import pexpect
class REPLWrapper(object):
    """Wrapper for a REPL.

    :param cmd_or_spawn: This can either be an instance of :class:`pexpect.spawn`
      in which a REPL has already been started, or a str command to start a new
      REPL process.
    :param str orig_prompt: The prompt to expect at first.
    :param str prompt_change: A command to change the prompt to something more
      unique. If this is ``None``, the prompt will not be changed. This will
      be formatted with the new and continuation prompts as positional
      parameters, so you can use ``{}`` style formatting to insert them into
      the command.
    :param str new_prompt: The more unique prompt to expect after the change.
    :param str extra_init_cmd: Commands to do extra initialisation, such as
      disabling pagers.
    """

    def __init__(self, cmd_or_spawn, orig_prompt, prompt_change, new_prompt=PEXPECT_PROMPT, continuation_prompt=PEXPECT_CONTINUATION_PROMPT, extra_init_cmd=None):
        if isinstance(cmd_or_spawn, basestring):
            self.child = pexpect.spawn(cmd_or_spawn, echo=False, encoding='utf-8')
        else:
            self.child = cmd_or_spawn
        if self.child.echo:
            self.child.setecho(False)
            self.child.waitnoecho()
        if prompt_change is None:
            self.prompt = orig_prompt
        else:
            self.set_prompt(orig_prompt, prompt_change.format(new_prompt, continuation_prompt))
            self.prompt = new_prompt
        self.continuation_prompt = continuation_prompt
        self._expect_prompt()
        if extra_init_cmd is not None:
            self.run_command(extra_init_cmd)

    def set_prompt(self, orig_prompt, prompt_change):
        self.child.expect(orig_prompt)
        self.child.sendline(prompt_change)

    def _expect_prompt(self, timeout=-1, async_=False):
        return self.child.expect_exact([self.prompt, self.continuation_prompt], timeout=timeout, async_=async_)

    def run_command(self, command, timeout=-1, async_=False):
        """Send a command to the REPL, wait for and return output.

        :param str command: The command to send. Trailing newlines are not needed.
          This should be a complete block of input that will trigger execution;
          if a continuation prompt is found after sending input, :exc:`ValueError`
          will be raised.
        :param int timeout: How long to wait for the next prompt. -1 means the
          default from the :class:`pexpect.spawn` object (default 30 seconds).
          None means to wait indefinitely.
        :param bool async_: On Python 3.4, or Python 3.3 with asyncio
          installed, passing ``async_=True`` will make this return an
          :mod:`asyncio` Future, which you can yield from to get the same
          result that this method would normally give directly.
        """
        cmdlines = command.splitlines()
        if command.endswith('\n'):
            cmdlines.append('')
        if not cmdlines:
            raise ValueError('No command was given')
        if async_:
            from ._async import repl_run_command_async
            return repl_run_command_async(self, cmdlines, timeout)
        res = []
        self.child.sendline(cmdlines[0])
        for line in cmdlines[1:]:
            self._expect_prompt(timeout=timeout)
            res.append(self.child.before)
            self.child.sendline(line)
        if self._expect_prompt(timeout=timeout) == 1:
            self.child.kill(signal.SIGINT)
            self._expect_prompt(timeout=1)
            raise ValueError('Continuation prompt found - input was incomplete:\n' + command)
        return u''.join(res + [self.child.before])