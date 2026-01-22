import os
from packaging.version import Version, parse
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, isdefined, PackageInfo
@staticmethod
def _format_xarray(val):
    """Convenience method for converting input arrays [1,2,3] to
        commandline format '1x2x3'"""
    return 'x'.join([str(x) for x in val])