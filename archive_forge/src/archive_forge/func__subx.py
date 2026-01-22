import enum
from . import _compiler, _parser
import functools
import copyreg
def _subx(pattern, template):
    template = _compile_repl(template, pattern)
    if not template[0] and len(template[1]) == 1:
        return template[1][0]

    def filter(match, template=template):
        return _parser.expand_template(template, match)
    return filter