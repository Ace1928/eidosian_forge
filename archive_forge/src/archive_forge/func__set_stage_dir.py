import os
import sys
import tempfile
import time
import click
import wandb
from wandb import env
def _set_stage_dir(stage_dir):
    global __stage_dir__
    __stage_dir__ = stage_dir