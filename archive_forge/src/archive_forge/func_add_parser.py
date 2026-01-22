import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def add_parser(self, name, **kwargs):
    if kwargs.get('prog') is None:
        kwargs['prog'] = '%s %s' % (self._prog_prefix, name)
    aliases = kwargs.pop('aliases', ())
    if 'help' in kwargs:
        help = kwargs.pop('help')
        choice_action = self._ChoicesPseudoAction(name, aliases, help)
        self._choices_actions.append(choice_action)
    parser = self._parser_class(**kwargs)
    self._name_parser_map[name] = parser
    for alias in aliases:
        self._name_parser_map[alias] = parser
    return parser