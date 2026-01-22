import inspect
import unittest
from traits.api import (
class OldCallable(BaseCallable):
    """
    Old-style Callable, whose validation tuple doesn't include
    the allow_none field.

    We only care about this case because it's possible that old pickles
    could include Callable instances whose validation tuple has length 1.
    """

    def __init__(self, value=None, **metadata):
        self.fast_validate = (ValidateTrait.callable,)
        super().__init__(value, **metadata)