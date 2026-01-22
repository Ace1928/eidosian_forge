from collections import namedtuple
import re
import textwrap
import warnings
def _list_0_or_more__compiled_re(element_re):
    return re.compile('^(?:$)|' + '(?:' + '(?:,|(?:' + element_re + '))' + '(?:' + OWS_re + ',(?:' + OWS_re + element_re + ')?)*' + ')$')