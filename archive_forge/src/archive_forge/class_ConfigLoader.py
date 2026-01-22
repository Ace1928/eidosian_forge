from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
class ConfigLoader:
    """A object for loading configurations from just about anywhere.

    The resulting configuration is packaged as a :class:`Config`.

    Notes
    -----
    A :class:`ConfigLoader` does one thing: load a config from a source
    (file, command line arguments) and returns the data as a :class:`Config` object.
    There are lots of things that :class:`ConfigLoader` does not do.  It does
    not implement complex logic for finding config files.  It does not handle
    default values or merge multiple configs.  These things need to be
    handled elsewhere.
    """

    def _log_default(self) -> Logger:
        from traitlets.log import get_logger
        return t.cast(Logger, get_logger())

    def __init__(self, log: Logger | None=None) -> None:
        """A base class for config loaders.

        log : instance of :class:`logging.Logger` to use.
              By default logger of :meth:`traitlets.config.application.Application.instance()`
              will be used

        Examples
        --------
        >>> cl = ConfigLoader()
        >>> config = cl.load_config()
        >>> config
        {}
        """
        self.clear()
        if log is None:
            self.log = self._log_default()
            self.log.debug('Using default logger')
        else:
            self.log = log

    def clear(self) -> None:
        self.config = Config()

    def load_config(self) -> Config:
        """Load a config from somewhere, return a :class:`Config` instance.

        Usually, this will cause self.config to be set and then returned.
        However, in most cases, :meth:`ConfigLoader.clear` should be called
        to erase any previous state.
        """
        self.clear()
        return self.config