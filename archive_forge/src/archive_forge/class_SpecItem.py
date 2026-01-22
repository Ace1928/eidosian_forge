import functools
import re
import warnings
class SpecItem(object):
    """A requirement specification."""
    KIND_ANY = '*'
    KIND_LT = '<'
    KIND_LTE = '<='
    KIND_EQUAL = '=='
    KIND_SHORTEQ = '='
    KIND_EMPTY = ''
    KIND_GTE = '>='
    KIND_GT = '>'
    KIND_NEQ = '!='
    KIND_CARET = '^'
    KIND_TILDE = '~'
    KIND_COMPATIBLE = '~='
    KIND_ALIASES = {KIND_SHORTEQ: KIND_EQUAL, KIND_EMPTY: KIND_EQUAL}
    re_spec = re.compile('^(<|<=||=|==|>=|>|!=|\\^|~|~=)(\\d.*)$')

    def __init__(self, requirement_string, _warn=True):
        if _warn:
            warnings.warn('The `SpecItem` class will be removed in 3.0.', DeprecationWarning, stacklevel=2)
        kind, spec = self.parse(requirement_string)
        self.kind = kind
        self.spec = spec
        self._clause = Spec(requirement_string).clause

    @classmethod
    def parse(cls, requirement_string):
        if not requirement_string:
            raise ValueError('Invalid empty requirement specification: %r' % requirement_string)
        if requirement_string == '*':
            return (cls.KIND_ANY, '')
        match = cls.re_spec.match(requirement_string)
        if not match:
            raise ValueError('Invalid requirement specification: %r' % requirement_string)
        kind, version = match.groups()
        if kind in cls.KIND_ALIASES:
            kind = cls.KIND_ALIASES[kind]
        spec = Version(version, partial=True)
        if spec.build is not None and kind not in (cls.KIND_EQUAL, cls.KIND_NEQ):
            raise ValueError('Invalid requirement specification %r: build numbers have no ordering.' % requirement_string)
        return (kind, spec)

    @classmethod
    def from_matcher(cls, matcher):
        if matcher == Always():
            return cls('*', _warn=False)
        elif matcher == Never():
            return cls('<0.0.0-', _warn=False)
        elif isinstance(matcher, Range):
            return cls('%s%s' % (matcher.operator, matcher.target), _warn=False)

    def match(self, version):
        return self._clause.match(version)

    def __str__(self):
        return '%s%s' % (self.kind, self.spec)

    def __repr__(self):
        return '<SpecItem: %s %r>' % (self.kind, self.spec)

    def __eq__(self, other):
        if not isinstance(other, SpecItem):
            return NotImplemented
        return self.kind == other.kind and self.spec == other.spec

    def __hash__(self):
        return hash((self.kind, self.spec))