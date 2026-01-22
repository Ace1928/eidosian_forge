import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
def _read_subprocess_stream(f, output_file, is_stdout=False):
    """Read and process a subprocess output stream.

    The goal is to find error messages and respond to them in a clever way.
    Currently just used for SSH messages (CONN_REFUSED, TIMEOUT, etc.), so
    the user does not get confused by these.

    Ran in a thread each for both `stdout` and `stderr` to
    allow for cross-platform asynchronous IO.

    Note: `select`-based IO is another option, but Windows has
    no support for `select`ing pipes, and Linux support varies somewhat.
    Spefically, Older *nix systems might also have quirks in how they
    handle `select` on pipes.

    Args:
        f: File object for the stream.
        output_file: File object to which filtered output is written.
        is_stdout (bool):
            When `is_stdout` is `False`, the stream is assumed to
            be `stderr`. Different error message detectors are used,
            and the output is displayed to the user unless it matches
            a special case (e.g. SSH timeout), in which case this is
            left up to the caller.
    """
    detected_special_case = None
    while True:
        line = f.readline()
        if line is None or line == '':
            break
        if line[-1] == '\n':
            line = line[:-1]
        if not is_stdout:
            if _ssh_output_regexes['connection_closed'].fullmatch(line) is not None:
                continue
            if _ssh_output_regexes['timeout'].fullmatch(line) is not None:
                if detected_special_case is not None:
                    raise ValueError('Bug: ssh_timeout conflicts with another special codition: ' + detected_special_case)
                detected_special_case = 'ssh_timeout'
                continue
            if _ssh_output_regexes['conn_refused'].fullmatch(line) is not None:
                if detected_special_case is not None:
                    raise ValueError('Bug: ssh_conn_refused conflicts with another special codition: ' + detected_special_case)
                detected_special_case = 'ssh_conn_refused'
                continue
            if _ssh_output_regexes['known_host_update'].fullmatch(line) is not None:
                continue
            cli_logger.error(line)
        if output_file is not None and output_file != subprocess.DEVNULL:
            output_file.write(line + '\n')
    return detected_special_case