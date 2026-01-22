from collections import defaultdict
from ..core import Store
class KeywordSettings:
    """
    Base class for options settings used to specified collections of
    keyword options.
    """
    allowed = {}
    defaults = dict([])
    options = dict(defaults.items())
    custom_exceptions = {}
    hidden = {}

    @classmethod
    def update_options(cls, options, items):
        """
        Allows updating options depending on class attributes
        and unvalidated options.
        """

    @classmethod
    def get_options(cls, items, options, warnfn):
        """Given a keyword specification, validate and compute options"""
        options = cls.update_options(options, items)
        for keyword in cls.defaults:
            if keyword in items:
                value = items[keyword]
                allowed = cls.allowed[keyword]
                if isinstance(allowed, set):
                    pass
                elif isinstance(allowed, dict):
                    if not isinstance(value, dict):
                        raise ValueError(f'Value {value!r} not a dict type')
                    disallowed = set(value.keys()) - set(allowed.keys())
                    if disallowed:
                        raise ValueError(f'Keywords {disallowed!r} for {keyword!r} option not one of {allowed}')
                    wrong_type = {k: v for k, v in value.items() if not isinstance(v, allowed[k])}
                    if wrong_type:
                        errors = []
                        for k, v in wrong_type.items():
                            errors.append(f"Value {v!r} for {keyword!r} option's {k!r} attribute not of type {allowed[k]!r}")
                        raise ValueError('\n'.join(errors))
                elif isinstance(allowed, list) and value not in allowed:
                    if keyword in cls.custom_exceptions:
                        cls.custom_exceptions[keyword](value, keyword, allowed)
                    else:
                        raise ValueError(f'Value {value!r} for key {keyword!r} not one of {allowed}')
                elif isinstance(allowed, tuple):
                    if not allowed[0] <= value <= allowed[1]:
                        info = (keyword, value) + allowed
                        raise ValueError('Value {!r} for key {!r} not between {} and {}'.format(*info))
                options[keyword] = value
        return cls._validate(options, items, warnfn)

    @classmethod
    def _validate(cls, options, items, warnfn):
        """Allows subclasses to check options are valid."""
        raise NotImplementedError('KeywordSettings is an abstract base class.')

    @classmethod
    def extract_keywords(cls, line, items):
        """
        Given the keyword string, parse a dictionary of options.
        """
        unprocessed = list(reversed(line.split('=')))
        while unprocessed:
            chunk = unprocessed.pop()
            key = None
            if chunk.strip() in cls.allowed:
                key = chunk.strip()
            else:
                raise SyntaxError(f'Invalid keyword: {chunk.strip()}')
            value = unprocessed.pop().strip()
            if len(unprocessed) != 0:
                for option in cls.allowed:
                    if value.endswith(option):
                        value = value[:-len(option)].strip()
                        unprocessed.append(option)
                        break
                else:
                    raise SyntaxError(f'Invalid keyword: {value.split()[-1]}')
            keyword = f'{key}={value}'
            try:
                items.update(eval(f'dict({keyword})'))
            except Exception:
                raise SyntaxError(f'Could not evaluate keyword: {keyword}') from None
        return items