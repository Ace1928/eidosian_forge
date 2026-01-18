import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def user_name_for(name):
    """ Returns a "user-friendly" version of a string, with the first letter
    capitalized and with underscore characters replaced by spaces. For example,
    ``user_name_for('user_name_for')`` returns ``'User name for'``.
    """
    name = name.replace('_', ' ')
    result = ''
    last_lower = False
    for c in name:
        if c.isupper() and last_lower:
            result += ' '
        last_lower = c.islower()
        result += c
    return result.capitalize()