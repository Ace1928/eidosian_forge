import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def _get_r_cmd_config(r_home: str, about: str, allow_empty=False):
    """Get the output of calling 'R CMD CONFIG <about>'.

    :param r_home: R HOME directory
    :param about: argument passed to the command line 'R CMD CONFIG'
    :param allow_empty: allow the output to be empty
    :return: a tuple (lines of output)"""
    r_exec = get_r_exec(r_home)
    cmd = (r_exec, 'CMD', 'config', about)
    logger.debug('Looking for R CONFIG with: {}'.format(' '.join(cmd)))
    output = subprocess.check_output(cmd, universal_newlines=True).split(os.linesep)
    if output[0].startswith('WARNING'):
        msg = 'R emitting a warning: {}'.format(output[0])
        warnings.warn(msg)
        logger.debug(msg)
        res = output[1:]
    else:
        res = output
    logger.debug(res)
    return res