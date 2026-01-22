import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
def _run_and_process_output(cmd, stdout_file, process_runner=subprocess, stderr_file=None, use_login_shells=False):
    """Run a command and process its output for special cases.

    Calls a standard 'check_call' if process_runner is not subprocess.

    Specifically, run all command output through regex to detect
    error conditions and filter out non-error messages that went to stderr
    anyway (SSH writes ALL of its "system" messages to stderr even if they
    are not actually errors).

    Args:
        cmd (List[str]): Command to run.
        process_runner: Used for command execution. Assumed to have
            'check_call' and 'check_output' inplemented.
        stdout_file: File to redirect stdout to.
        stderr_file: File to redirect stderr to.

    Implementation notes:
    1. `use_login_shells` disables special processing
    If we run interactive apps, output processing will likely get
    overwhelmed with the interactive output elements.
    Thus, we disable output processing for login shells. This makes
    the logging experience considerably worse, but it only degrades
    to old-style logging.

    For example, `pip install` outputs HUNDREDS of progress-bar lines
    when downloading a package, and we have to
    read + regex + write all of them.

    After all, even just printing output to console can often slow
    down a fast-printing app, and we do more than just print, and
    all that from Python, which is much slower than C regarding
    stream processing.

    2. `stdin=PIPE` for subprocesses
    Do not inherit stdin as it messes with bash signals
    (ctrl-C for SIGINT) and these commands aren't supposed to
    take input anyway.

    3. `ThreadPoolExecutor` without the `Pool`
    We use `ThreadPoolExecutor` to create futures from threads.
    Threads are never reused.

    This approach allows us to have no custom synchronization by
    off-loading the return value and exception passing to the
    standard library (`ThreadPoolExecutor` internals).

    This instance will be `shutdown()` ASAP so it's fine to
    create one in such a weird place.

    The code is thus 100% thread-safe as long as the stream readers
    are read-only except for return values and possible exceptions.
    """
    stdin_overwrite = subprocess.PIPE
    assert not (does_allow_interactive() and is_output_redirected()), 'Cannot redirect output while in interactive mode.'
    if process_runner != subprocess or (does_allow_interactive() and (not is_output_redirected())):
        stdin_overwrite = None
    if use_login_shells or process_runner != subprocess:
        return process_runner.check_call(cmd, stdin=stdin_overwrite, stdout=stdout_file, stderr=stderr_file)
    with subprocess.Popen(cmd, stdin=stdin_overwrite, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        from concurrent.futures import ThreadPoolExecutor
        p.stdin.close()
        with ThreadPoolExecutor(max_workers=2) as executor:
            stdout_future = executor.submit(_read_subprocess_stream, p.stdout, stdout_file, is_stdout=True)
            stderr_future = executor.submit(_read_subprocess_stream, p.stderr, stderr_file, is_stdout=False)
            executor.shutdown()
            p.poll()
            detected_special_case = stdout_future.result()
            if stderr_future.result() is not None:
                if detected_special_case is not None:
                    raise ValueError('Bug: found a special case in both stdout and stderr. This is not valid behavior at the time of writing this code.')
                detected_special_case = stderr_future.result()
            if p.returncode > 0:
                raise ProcessRunnerError('Command failed', 'ssh_command_failed', code=p.returncode, command=cmd, special_case=detected_special_case)
            elif p.returncode < 0:
                raise ProcessRunnerError('Command failed', 'ssh_command_failed', code=p.returncode, command=cmd, special_case='died_to_signal')
            return p.returncode