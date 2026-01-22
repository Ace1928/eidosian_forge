from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six.moves.collections_abc import MutableMapping
def _snake_to_camel(snake, capitalize_first=False):
    if capitalize_first:
        return ''.join((x.capitalize() or '_' for x in snake.split('_')))
    else:
        return snake.split('_')[0] + ''.join((x.capitalize() or '_' for x in snake.split('_')[1:]))