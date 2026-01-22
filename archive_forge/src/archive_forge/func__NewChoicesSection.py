from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _NewChoicesSection(name, choices):
    return _CreateItem('{name} is one of the following:'.format(name=formatting.Bold(formatting.Underline(name))), '\n' + '\n\n'.join(choices), indent=1)