import configparser
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Union
from .helpers import make_path
def _build_getter(cfg_obj, cfg_section, method, converter=None):

    def caller(option, **kwargs):
        try:
            rv = getattr(cfg_obj, method)(cfg_section, option, **kwargs)
        except configparser.NoSectionError as nse:
            raise MissingConfigSection(f'No config section named {cfg_section}') from nse
        except configparser.NoOptionError as noe:
            raise MissingConfigItem(f'No config item for {option}') from noe
        except ValueError as ve:
            raise ConfigValueTypeError(f'Wrong value type for {option}') from ve
        else:
            if converter:
                try:
                    rv = converter(rv)
                except Exception as e:
                    raise ConfigValueTypeError(f'Wrong value type for {option}') from e
            return rv
    return caller