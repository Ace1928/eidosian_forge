import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
class Termcap(object):
    """Terminal capability of given variable name and pattern."""

    def __init__(self, name, pattern, attribute):
        """
        Class initializer.

        :arg str name: name describing capability.
        :arg str pattern: regular expression string.
        :arg str attribute: :class:`~.Terminal` attribute used to build
            this terminal capability.
        """
        self.name = name
        self.pattern = pattern
        self.attribute = attribute
        self._re_compiled = None

    def __repr__(self):
        return '<Termcap {self.name}:{self.pattern!r}>'.format(self=self)

    @property
    def named_pattern(self):
        """Regular expression pattern for capability with named group."""
        return '(?P<{self.name}>{self.pattern})'.format(self=self)

    @property
    def re_compiled(self):
        """Compiled regular expression pattern for capability."""
        if self._re_compiled is None:
            self._re_compiled = re.compile(self.pattern)
        return self._re_compiled

    @property
    def will_move(self):
        """Whether capability causes cursor movement."""
        return self.name in CAPABILITIES_CAUSE_MOVEMENT

    def horizontal_distance(self, text):
        """
        Horizontal carriage adjusted by capability, may be negative.

        :rtype: int
        :arg str text: for capabilities *parm_left_cursor*,
            *parm_right_cursor*, provide the matching sequence
            text, its interpreted distance is returned.

        :returns: 0 except for matching '
        """
        value = {'cursor_left': -1, 'backspace': -1, 'cursor_right': 1, 'tab': 8, 'ascii_tab': 8}.get(self.name)
        if value is not None:
            return value
        unit = {'parm_left_cursor': -1, 'parm_right_cursor': 1}.get(self.name)
        if unit is not None:
            value = int(self.re_compiled.match(text).group(1))
            return unit * value
        return 0

    @classmethod
    def build(cls, name, capability, attribute, nparams=0, numeric=99, match_grouped=False, match_any=False, match_optional=False):
        """
        Class factory builder for given capability definition.

        :arg str name: Variable name given for this pattern.
        :arg str capability: A unicode string representing a terminal
            capability to build for. When ``nparams`` is non-zero, it
            must be a callable unicode string (such as the result from
            ``getattr(term, 'bold')``.
        :arg str attribute: The terminfo(5) capability name by which this
            pattern is known.
        :arg int nparams: number of positional arguments for callable.
        :arg int numeric: Value to substitute into capability to when generating pattern
        :arg bool match_grouped: If the numeric pattern should be
            grouped, ``(\\d+)`` when ``True``, ``\\d+`` default.
        :arg bool match_any: When keyword argument ``nparams`` is given,
            *any* numeric found in output is suitable for building as
            pattern ``(\\d+)``.  Otherwise, only the first matching value of
            range *(numeric - 1)* through *(numeric + 1)* will be replaced by
            pattern ``(\\d+)`` in builder.
        :arg bool match_optional: When ``True``, building of numeric patterns
            containing ``(\\d+)`` will be built as optional, ``(\\d+)?``.
        :rtype: blessed.sequences.Termcap
        :returns: Terminal capability instance for given capability definition
        """
        _numeric_regex = '\\d+'
        if match_grouped:
            _numeric_regex = '(\\d+)'
        if match_optional:
            _numeric_regex = '(\\d+)?'
        numeric = 99 if numeric is None else numeric
        if nparams == 0:
            return cls(name, re.escape(capability), attribute)
        _outp = re.escape(capability(*(numeric,) * nparams))
        if not match_any:
            for num in range(numeric - 1, numeric + 2):
                if str(num) in _outp:
                    pattern = _outp.replace(str(num), _numeric_regex)
                    return cls(name, pattern, attribute)
        if match_grouped:
            pattern = re.sub('(\\d+)', lambda x: _numeric_regex, _outp)
        else:
            pattern = re.sub('\\d+', lambda x: _numeric_regex, _outp)
        return cls(name, pattern, attribute)