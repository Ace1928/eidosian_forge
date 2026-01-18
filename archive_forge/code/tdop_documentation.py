import sys
import re
from abc import ABCMeta
from unicodedata import name as unicode_name
from decimal import Decimal, DecimalException
from typing import Any, cast, overload, Callable, Dict, Generic, List, \

        Returns a regex based tokenizer built from a symbol table of token classes.
        The returned tokenizer skips extra spaces between symbols.

        A regular expression is created from the symbol table of the parser using a template.
        The symbols are inserted in the template putting the longer symbols first. Symbols and
        their patterns can't contain spaces.

        :param symbol_table: a dictionary containing the token classes of the formal language.
        