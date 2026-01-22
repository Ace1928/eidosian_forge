import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
import logging
def _format_failed_pyspy_command(cmd, stdout, stderr) -> str:
    stderr_str = stderr.decode('utf-8')
    extra_message = ''
    if 'permission' in stderr_str.lower():
        set_chown_command = DARWIN_SET_CHOWN_CMD if sys.platform == 'darwin' else LINUX_SET_CHOWN_CMD
        extra_message = PYSPY_PERMISSIONS_ERROR_MESSAGE.format(set_chown_command=set_chown_command)
    return f'Failed to execute `{cmd}`.\n{extra_message}\n=== stderr ===\n{stderr.decode('utf-8')}\n\n=== stdout ===\n{stdout.decode('utf-8')}\n'