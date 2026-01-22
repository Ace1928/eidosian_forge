import io
import sys
import traceback
from . import util
from functools import wraps
def _restoreStdout(self):
    if self.buffer:
        if self._mirrorOutput:
            output = sys.stdout.getvalue()
            error = sys.stderr.getvalue()
            if output:
                if not output.endswith('\n'):
                    output += '\n'
                self._original_stdout.write(STDOUT_LINE % output)
            if error:
                if not error.endswith('\n'):
                    error += '\n'
                self._original_stderr.write(STDERR_LINE % error)
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._stdout_buffer.seek(0)
        self._stdout_buffer.truncate()
        self._stderr_buffer.seek(0)
        self._stderr_buffer.truncate()