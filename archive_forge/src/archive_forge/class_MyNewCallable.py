import inspect
import unittest
from traits.api import (
class MyNewCallable(HasTraits):
    value = Callable(default_value=pow, allow_none=False)