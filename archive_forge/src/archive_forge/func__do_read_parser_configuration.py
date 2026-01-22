import argparse
import contextlib
import os
import sys
from configparser import ConfigParser
from typing import Dict, List, Union
from docformatter import __pkginfo__
def _do_read_parser_configuration(self) -> None:
    """Load configuration information from a *.cfg or *.ini file."""
    config = ConfigParser()
    config.read(self.config_file)
    for _section in ['tool.docformatter', 'tool:docformatter', 'docformatter']:
        if _section in config.sections():
            self.flargs_dct = {k: v if isinstance(v, list) else str(v) for k, v in config[_section].items()}