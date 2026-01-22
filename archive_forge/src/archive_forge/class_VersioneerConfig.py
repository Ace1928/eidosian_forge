import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
class VersioneerConfig:
    """Container for Versioneer configuration parameters."""