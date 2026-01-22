import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def _call_new_python(self, context, *py_args, **kwargs):
    """Executes the newly created Python using safe-ish options"""
    args = [context.env_exec_cmd, *py_args]
    kwargs['env'] = env = os.environ.copy()
    env['VIRTUAL_ENV'] = context.env_dir
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    kwargs['cwd'] = context.env_dir
    kwargs['executable'] = context.env_exec_cmd
    subprocess.check_output(args, **kwargs)