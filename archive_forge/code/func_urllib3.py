import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_SYNC)
def urllib3(session):
    session.install(LIBRARY_DIR)
    session.install(*TEST_DEPENDENCIES_SYNC)
    session.env[EXPLICIT_CREDENTIALS_ENV] = SERVICE_ACCOUNT_FILE
    default(session, 'system_tests_sync/test_urllib3.py', *session.posargs)