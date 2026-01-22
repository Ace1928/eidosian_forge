from __future__ import annotations
import copy
import os
import secrets
import threading
from collections import OrderedDict
from typing import Any, Callable
from blinker import Signal
from streamlit import config_util, development, env_util, file_util, util
from streamlit.config_option import ConfigOption
from streamlit.errors import StreamlitAPIException
def _create_option(key: str, description: str | None=None, default_val: Any | None=None, scriptable: bool=False, visibility: str='visible', deprecated: bool=False, deprecation_text: str | None=None, expiration_date: str | None=None, replaced_by: str | None=None, type_: type=str, sensitive: bool=False) -> ConfigOption:
    '''Create a ConfigOption and store it globally in this module.

    There are two ways to create a ConfigOption:

        (1) Simple, constant config options are created as follows:

            _create_option('section.optionName',
                description = 'Put the description here.',
                default_val = 12345)

        (2) More complex, programmable config options use decorator syntax to
        resolve their values at runtime:

            @_create_option('section.optionName')
            def _section_option_name():
                """Put the description here."""
                return 12345

    To achieve this sugar, _create_option() returns a *callable object* of type
    ConfigObject, which then decorates the function.

    NOTE: ConfigObjects call their evaluation functions *every time* the option
    is requested. To prevent this, use the `streamlit.util.memoize` decorator as
    follows:

            @_create_option('section.memoizedOptionName')
            @util.memoize
            def _section_memoized_option_name():
                """Put the description here."""

                (This function is only called once.)
                """
                return 12345

    '''
    option = ConfigOption(key, description=description, default_val=default_val, scriptable=scriptable, visibility=visibility, deprecated=deprecated, deprecation_text=deprecation_text, expiration_date=expiration_date, replaced_by=replaced_by, type_=type_, sensitive=sensitive)
    assert option.section in _section_descriptions, 'Section "{}" must be one of {}.'.format(option.section, ', '.join(_section_descriptions.keys()))
    assert key not in _config_options_template, 'Cannot define option "%s" twice.' % key
    _config_options_template[key] = option
    return option