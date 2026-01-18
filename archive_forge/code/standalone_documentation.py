from copy import deepcopy
from abc import ABC, abstractmethod
from types import ModuleType
from typing import (
import sys
import token, tokenize
import os
from os import path
from collections import defaultdict
from functools import partial
from argparse import ArgumentParser
import lark
from lark.tools import lalr_argparser, build_lalr, make_warnings_comments
from lark.grammar import Rule
from lark.lexer import TerminalDef
 Strip comments and docstrings from a file.
    Based on code from: https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings
    