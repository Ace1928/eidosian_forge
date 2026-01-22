import os
import inspect
import subprocess
def _get_git_revision():
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    return revision.decode('utf-8')