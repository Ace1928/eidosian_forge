from .api import (
class CompoundValidator(FancyValidator):
    """Base class for all compound validators."""
    if_invalid = NoDefault
    accept_iterator = False
    validators = []
    __unpackargs__ = ('*', 'validatorArgs')
    __mutableattributes__ = ('validators',)
    _deprecated_methods = (('attempt_convert', '_attempt_convert'),)

    @staticmethod
    def __classinit__(cls, new_attrs):
        FancyValidator.__classinit__(cls, new_attrs)
        to_add = []
        for name, value in new_attrs.items():
            if is_validator(value) and value is not Identity:
                to_add.append((name, value))
                delattr(cls, name)
        to_add.sort()
        cls.validators.extend([value for _name, value in to_add])

    def __init__(self, *args, **kw):
        Validator.__init__(self, *args, **kw)
        self.validators = self.validators[:]
        self.validators.extend(self.validatorArgs)

    @staticmethod
    def _repr_vars(names):
        return [n for n in Validator._repr_vars(names) if n != 'validatorArgs']

    def _attempt_convert(self, value, state, convertFunc):
        raise NotImplementedError('Subclasses must implement _attempt_convert')

    def _convert_to_python(self, value, state=None):
        return self._attempt_convert(value, state, to_python)

    def _convert_from_python(self, value, state=None):
        return self._attempt_convert(value, state, from_python)

    def subvalidators(self):
        return self.validators