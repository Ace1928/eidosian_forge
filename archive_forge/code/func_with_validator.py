from .api import (
def with_validator(self, validator):
    """Add another validator.

        Adds the validator (or list of validators) to a copy of
        this validator.
        """
    new = self.validators[:]
    if isinstance(validator, (list, tuple)):
        new.extend(validator)
    else:
        new.append(validator)
    return self.__class__(*new, **dict(if_invalid=self.if_invalid))