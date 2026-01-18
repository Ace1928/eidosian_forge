from __future__ import annotations
import os
import re
import subprocess
import typing as T
from .. import mlog
from .. import mesonlib
from ..compilers.compilers import CrossNoRunException
from ..mesonlib import (
from ..environment import detect_cpu_family
from .base import DependencyException, DependencyMethods, DependencyTypeName, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
gnustep-config returns a bunch of garbage args such as -O2 and so
        on. Drop everything that is not needed.
        