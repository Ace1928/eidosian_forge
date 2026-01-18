from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple
import pygit2
Set the name and email for the pygit repo collecting from the gitconfig.
        If not available in gitconfig, set the values from the passed arguments.