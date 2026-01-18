import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def update_cfg(cfgp, config_args):
    for arg in config_args:
        try:
            lhs, rhs = arg.split('=', maxsplit=1)
            section, item = lhs.split('.')
            if not cfgp.has_section(section):
                cfgp.add_section(section)
            cfgp.set(section, item, rhs)
        except Exception:
            pass