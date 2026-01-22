import logging
import os
import shutil
import subprocess
import hashlib
import json
from typing import Optional, List, Union, Tuple
Runs a command as a child process, streaming output to the logger.

    The last n_lines lines of output are also returned (stdout and stderr).
    