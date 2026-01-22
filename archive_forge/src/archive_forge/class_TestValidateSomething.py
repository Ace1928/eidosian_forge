import os
import pytest
class TestValidateSomething(ValidateAPI):
    """Example implementing an API validator test class"""

    def obj_params(self):
        """Iterator returning (obj, params) pairs

        ``obj`` is some instance for which we want to check the API.

        ``params`` is a mapping with parameters that you are going to check
        against ``obj``.  See the :meth:`validate_something` method for an
        example.
        """

        class C:

            def __init__(self, var):
                self.var = var

            def get_var(self):
                return self.var
        yield (C(5), {'var': 5})
        yield (C('easypeasy'), {'var': 'easypeasy'})

    def validate_something(self, obj, params):
        """Do some checks of the `obj` API against `params`

        The metaclass sets up a ``test_something`` function that runs these
        checks on each (
        """
        assert obj.var == params['var']
        assert obj.get_var() == params['var']