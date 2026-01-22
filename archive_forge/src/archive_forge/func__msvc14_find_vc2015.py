import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
def _msvc14_find_vc2015():
    """Python 3.8 "distutils/_msvccompiler.py" backport"""
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'Software\\Microsoft\\VisualStudio\\SxS\\VC7', 0, winreg.KEY_READ | winreg.KEY_WOW64_32KEY)
    except OSError:
        return (None, None)
    best_version = 0
    best_dir = None
    with key:
        for i in itertools.count():
            try:
                v, vc_dir, vt = winreg.EnumValue(key, i)
            except OSError:
                break
            if v and vt == winreg.REG_SZ and isdir(vc_dir):
                try:
                    version = int(float(v))
                except (ValueError, TypeError):
                    continue
                if version >= 14 and version > best_version:
                    best_version, best_dir = (version, vc_dir)
    return (best_version, best_dir)