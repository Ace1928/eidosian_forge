import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
class InvalidTransitionException(Exception):
    pass