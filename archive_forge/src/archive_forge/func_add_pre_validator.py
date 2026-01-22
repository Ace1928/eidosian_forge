import warnings
from .api import _, is_validator, FancyValidator, Invalid, NoDefault
from . import declarative
from .exc import FERuntimeWarning
@declarative.classinstancemethod
def add_pre_validator(self, cls, validator):
    if self is not None:
        if self.pre_validators is cls.pre_validators:
            self.pre_validators = cls.pre_validators[:]
        self.pre_validators.append(validator)
    else:
        cls.pre_validators.append(validator)