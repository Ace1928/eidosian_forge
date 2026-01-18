import os
import threading
import subprocess
import sys
import time
import signal
import shlex
from .spawnbase import SpawnBase, PY3
from .exceptions import EOF
from .utils import string_types
Closes the stdin pipe from the writing end.