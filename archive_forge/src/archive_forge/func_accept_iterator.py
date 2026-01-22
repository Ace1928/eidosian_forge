from .api import (
@property
def accept_iterator(self):
    accept_iterator = True
    for validator in self.validators:
        accept_iterator = accept_iterator and getattr(validator, 'accept_iterator', False)
    return accept_iterator