import os
from configparser import ConfigParser
from distutils import log
from distutils.command.build_py import build_py
from distutils.command.install_scripts import install_scripts
from distutils.version import LooseVersion
from os.path import join as pjoin
from os.path import split as psplit
from os.path import splitext
def _add_append_key(in_dict, key, value):
    """Helper for appending dependencies to setuptools args"""
    if key not in in_dict:
        in_dict[key] = []
    elif isinstance(in_dict[key], str):
        in_dict[key] = [in_dict[key]]
    in_dict[key].append(value)