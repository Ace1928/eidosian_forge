import os
import shutil
import subprocess
import sys
import fixtures
import testresources
import testtools
from testtools import content
from pbr import options
def _config_git():
    _run_cmd(['git', 'config', '--global', 'user.email', 'example@example.com'], None)
    _run_cmd(['git', 'config', '--global', 'user.name', 'OpenStack Developer'], None)
    _run_cmd(['git', 'config', '--global', 'user.signingkey', 'example@example.com'], None)