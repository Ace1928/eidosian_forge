from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
@staticmethod
def translate_schema(schema: dict) -> dict:
    """Translate a schema.

        Parameters
        ----------
        schema: dict
            The schema to be translated

        Returns
        -------
        Dict
            The translated schema
        """
    if translator._LOCALE == DEFAULT_LOCALE:
        return schema
    translations = translator.load(schema.get(_lab_i18n_config, {}).get('domain', DEFAULT_DOMAIN))
    new_schema = schema.copy()
    translator._translate_schema_strings(translations, new_schema)
    return new_schema