import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
class StrOpt(Opt):
    """Option with String type

    Option with ``type`` :class:`oslo_config.types.String`

    :param name: the option's name
    :param choices: Optional sequence of either valid values or tuples of valid
        values with descriptions.
    :param quotes: If True and string is enclosed with single or double
                   quotes, will strip those quotes.
    :param regex: Optional regular expression (string or compiled
                  regex) that the value must match on an unanchored
                  search.
    :param ignore_case: If True case differences (uppercase vs. lowercase)
                        between 'choices' or 'regex' will be ignored.
    :param max_length: If positive integer, the value must be less than or
                       equal to this parameter.
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionchanged:: 2.7
       Added *quotes* parameter

    .. versionchanged:: 2.7
       Added *regex* parameter

    .. versionchanged:: 2.7
       Added *ignore_case* parameter

    .. versionchanged:: 2.7
       Added *max_length* parameter

    .. versionchanged:: 5.2
       The *choices* parameter will now accept a sequence of tuples, where each
       tuple is of form (*choice*, *description*)
    """

    def __init__(self, name, choices=None, quotes=None, regex=None, ignore_case=False, max_length=None, **kwargs):
        super(StrOpt, self).__init__(name, type=types.String(choices=choices, quotes=quotes, regex=regex, ignore_case=ignore_case, max_length=max_length), **kwargs)

    def _get_choice_text(self, choice):
        if choice is None:
            return '<None>'
        elif choice == '':
            return "''"
        return str(choice)

    def _get_argparse_kwargs(self, group, **kwargs):
        """Extends the base argparse keyword dict for the config dir option."""
        kwargs = super(StrOpt, self)._get_argparse_kwargs(group)
        if getattr(self.type, 'choices', None):
            choices_text = ', '.join([self._get_choice_text(choice) for choice in self.type.choices])
            if kwargs['help'] is None:
                kwargs['help'] = ''
            kwargs['help'].rstrip('\n')
            kwargs['help'] += '\n Allowed values: %s\n' % choices_text
        return kwargs