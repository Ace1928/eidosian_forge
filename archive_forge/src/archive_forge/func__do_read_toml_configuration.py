import argparse
import contextlib
import os
import sys
from configparser import ConfigParser
from typing import Dict, List, Union
from docformatter import __pkginfo__
def _do_read_toml_configuration(self) -> None:
    """Load configuration information from a *.toml file."""
    with open(self.config_file, 'rb') as f:
        if TOMLI_INSTALLED:
            config = tomli.load(f)
        elif TOMLLIB_INSTALLED:
            config = tomllib.load(f)
    result = config.get('tool', {}).get('docformatter', None)
    if result is not None:
        self.flargs_dct = {k: v if isinstance(v, list) else str(v) for k, v in result.items()}