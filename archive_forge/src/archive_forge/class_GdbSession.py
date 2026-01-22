from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
class GdbSession:
    proc = None
    verbose = True

    def __init__(self, *args, **env):
        gdb_env = environment_for_gdb()
        gdb_env.update(env)
        self.proc = subprocess.Popen(gdb_command + list(args), env=gdb_env, bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.last_stdout = []
        self.last_stdout_line = b''

    def wait_until_ready(self):
        """
        Record output until the gdb prompt displays.  Return recorded output.
        """
        while not self.last_stdout_line.startswith(b'(gdb) ') and self.proc.poll() is None:
            block = self.proc.stdout.read(4096)
            if self.verbose:
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
            block, sep, last_line = block.rpartition(b'\n')
            if sep:
                self.last_stdout.append(self.last_stdout_line)
                self.last_stdout.append(block + sep)
                self.last_stdout_line = last_line
            else:
                assert block == b''
                self.last_stdout_line += last_line
        if self.proc.poll() is not None:
            raise IOError('gdb session terminated unexpectedly')
        out = b''.join(self.last_stdout).decode('utf-8')
        self.last_stdout = []
        self.last_stdout_line = b''
        return out

    def issue_command(self, line):
        line = line.encode('utf-8') + b'\n'
        if self.verbose:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def run_command(self, line):
        self.issue_command(line)
        return self.wait_until_ready()

    def print_value(self, expr):
        """
        Ask gdb to print the value of an expression and return the result.
        """
        out = self.run_command(f'p {expr}')
        out, n = re.subn('^\\$\\d+ = ', '', out)
        assert n == 1, out
        return out.strip()

    def select_frame(self, func_name):
        """
        Select the innermost frame with the given function name.
        """
        out = self.run_command('info stack')
        pat = '(?mi)^#(\\d+)\\s+.* in ' + re.escape(func_name) + '\\b'
        m = re.search(pat, out)
        if m is None:
            pytest.fail(f'Could not select frame for function {func_name}')
        frame_num = int(m[1])
        out = self.run_command(f'frame {frame_num}')
        assert f'in {func_name}' in out

    def join(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def __del__(self):
        self.join()