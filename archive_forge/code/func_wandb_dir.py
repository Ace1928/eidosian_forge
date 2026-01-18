import os
import sys
import tempfile
import time
import click
import wandb
from wandb import env
def wandb_dir(root_dir=None):
    if root_dir is None or root_dir == '':
        try:
            cwd = os.getcwd()
        except OSError:
            termwarn('os.getcwd() no longer exists, using system temp directory')
            cwd = tempfile.gettempdir()
        root_dir = env.get_dir(cwd)
    path = os.path.join(root_dir, __stage_dir__ or 'wandb' + os.sep)
    if not os.access(root_dir, os.W_OK):
        termwarn(f"Path {path} wasn't writable, using system temp directory", repeat=False)
        path = os.path.join(tempfile.gettempdir(), __stage_dir__ or 'wandb' + os.sep)
    return path