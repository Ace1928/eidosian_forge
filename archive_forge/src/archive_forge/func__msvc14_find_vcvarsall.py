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
def _msvc14_find_vcvarsall(plat_spec):
    """Python 3.8 "distutils/_msvccompiler.py" backport"""
    _, best_dir = _msvc14_find_vc2017()
    vcruntime = None
    if plat_spec in PLAT_SPEC_TO_RUNTIME:
        vcruntime_plat = PLAT_SPEC_TO_RUNTIME[plat_spec]
    else:
        vcruntime_plat = 'x64' if 'amd64' in plat_spec else 'x86'
    if best_dir:
        vcredist = join(best_dir, '..', '..', 'redist', 'MSVC', '**', vcruntime_plat, 'Microsoft.VC14*.CRT', 'vcruntime140.dll')
        try:
            import glob
            vcruntime = glob.glob(vcredist, recursive=True)[-1]
        except (ImportError, OSError, LookupError):
            vcruntime = None
    if not best_dir:
        best_version, best_dir = _msvc14_find_vc2015()
        if best_version:
            vcruntime = join(best_dir, 'redist', vcruntime_plat, 'Microsoft.VC140.CRT', 'vcruntime140.dll')
    if not best_dir:
        return (None, None)
    vcvarsall = join(best_dir, 'vcvarsall.bat')
    if not isfile(vcvarsall):
        return (None, None)
    if not vcruntime or not isfile(vcruntime):
        vcruntime = None
    return (vcvarsall, vcruntime)