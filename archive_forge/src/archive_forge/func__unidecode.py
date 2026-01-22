import warnings
from typing import Dict, Optional, Sequence
def _unidecode(string: str, errors: str, replace_str: str) -> str:
    retval = []
    for index, char in enumerate(string):
        repl = _get_repl_str(char)
        if repl is None:
            if errors == 'ignore':
                repl = ''
            elif errors == 'strict':
                raise UnidecodeError('no replacement found for character %r in position %d' % (char, index), index)
            elif errors == 'replace':
                repl = replace_str
            elif errors == 'preserve':
                repl = char
            else:
                raise UnidecodeError('invalid value for errors parameter %r' % (errors,))
        retval.append(repl)
    return ''.join(retval)