import io
import json
import os
import platform
import shutil
import subprocess
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from cmdstanpy.utils import get_logger
from cmdstanpy.utils.cmdstan import (
from cmdstanpy.utils.command import do_command
from cmdstanpy.utils.filesystem import SanitizedOrTmpFilePath
def validate_user_header(self) -> None:
    """
        User header exists.
        Raise ValueError if bad config is found.
        """
    if self._user_header != '':
        if not (os.path.exists(self._user_header) and os.path.isfile(self._user_header)):
            raise ValueError(f'User header file {self._user_header} cannot be found')
        if self._user_header[-4:] != '.hpp':
            raise ValueError(f'Header file must end in .hpp, got {self._user_header}')
        if 'allow-undefined' not in self._stanc_options:
            self._stanc_options['allow-undefined'] = True
        self._user_header = os.path.abspath(self._user_header)
        if ' ' in self._user_header:
            raise ValueError('User header must be in a location with no spaces in path!')
        if 'USER_HEADER' in self._cpp_options and self._user_header != self._cpp_options['USER_HEADER']:
            raise ValueError(f'Disagreement in user_header C++ options found!\n{self._user_header}, {self._cpp_options['USER_HEADER']}')
        self._cpp_options['USER_HEADER'] = self._user_header