import functools
import os
import subprocess
import sys
from mlflow.utils.os import is_windows
def _join_commands(*commands):
    entry_point = ['bash', '-c'] if not is_windows() else ['cmd', '/c']
    sep = ' && ' if not is_windows() else ' & '
    return [*entry_point, sep.join(map(str, commands))]