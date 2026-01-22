import configparser
import locale
import os
import sys
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple
from pip._internal.exceptions import (
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum
def _construct_parser(self, fname: str) -> RawConfigParser:
    parser = configparser.RawConfigParser()
    if os.path.exists(fname):
        locale_encoding = locale.getpreferredencoding(False)
        try:
            parser.read(fname, encoding=locale_encoding)
        except UnicodeDecodeError:
            raise ConfigurationFileCouldNotBeLoaded(reason=f'contains invalid {locale_encoding} characters', fname=fname)
        except configparser.Error as error:
            raise ConfigurationFileCouldNotBeLoaded(error=error)
    return parser