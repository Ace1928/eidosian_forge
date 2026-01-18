import os
from distutils.core import Command
from distutils.util import change_root, convert_path
distutils.command.install_data

Implements the Distutils 'install_data' command, for installing
platform-independent data files.