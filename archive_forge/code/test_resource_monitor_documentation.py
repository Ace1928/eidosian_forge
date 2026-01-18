import os
import pytest
from .... import config
from ....utils.profiler import _use_resources
from ...base import traits, CommandLine, CommandLineInputSpec
from ... import utility as niu

    Test runtime profiler correctly records workflow RAM/CPUs consumption
    of a Function interface
    