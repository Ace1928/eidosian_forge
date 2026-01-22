import configparser
import os
import shlex
import subprocess
from os.path import expanduser, expandvars
from pathlib import Path
from typing import List, Optional, Union
from gitlab.const import USER_AGENT
class GitlabDataError(ConfigError):
    pass