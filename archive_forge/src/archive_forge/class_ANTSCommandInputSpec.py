import os
from packaging.version import Version, parse
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, isdefined, PackageInfo
class ANTSCommandInputSpec(CommandLineInputSpec):
    """Base Input Specification for all ANTS Commands"""
    num_threads = traits.Int(LOCAL_DEFAULT_NUMBER_OF_THREADS, usedefault=True, nohash=True, desc='Number of ITK threads to use')