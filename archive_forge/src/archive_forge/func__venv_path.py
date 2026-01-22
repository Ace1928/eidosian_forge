import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def _venv_path(self, env_dir, name):
    vars = {'base': env_dir, 'platbase': env_dir, 'installed_base': env_dir, 'installed_platbase': env_dir}
    return sysconfig.get_path(name, scheme='venv', vars=vars)