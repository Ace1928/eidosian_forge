from __future__ import annotations
import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
from traitlets.traitlets import (
from traitlets.utils.bunch import Bunch
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs
from ..utils import cast_unicode
from ..utils.importstring import import_item
@classmethod
def _load_config_files(cls, basefilename: str, path: str | t.Sequence[str | None] | None, log: AnyLogger | None=None, raise_config_file_errors: bool=False) -> t.Generator[t.Any, None, None]:
    """Load config files (py,json) by filename and path.

        yield each config object in turn.
        """
    if isinstance(path, str) or path is None:
        path = [path]
    for current in reversed(path):
        pyloader = cls.python_config_loader_class(basefilename + '.py', path=current, log=log)
        if log:
            log.debug('Looking for %s in %s', basefilename, current or os.getcwd())
        jsonloader = cls.json_config_loader_class(basefilename + '.json', path=current, log=log)
        loaded: list[t.Any] = []
        filenames: list[str] = []
        for loader in [pyloader, jsonloader]:
            config = None
            try:
                config = loader.load_config()
            except ConfigFileNotFound:
                pass
            except Exception:
                filename = loader.full_filename or basefilename
                if raise_config_file_errors:
                    raise
                if log:
                    log.error('Exception while loading config file %s', filename, exc_info=True)
            else:
                if log:
                    log.debug('Loaded config file: %s', loader.full_filename)
            if config:
                for filename, earlier_config in zip(filenames, loaded):
                    collisions = earlier_config.collisions(config)
                    if collisions and log:
                        log.warning('Collisions detected in {0} and {1} config files. {1} has higher priority: {2}'.format(filename, loader.full_filename, json.dumps(collisions, indent=2)))
                yield (config, loader.full_filename)
                loaded.append(config)
                filenames.append(loader.full_filename)