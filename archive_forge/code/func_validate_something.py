import os
import pytest
def validate_something(self, obj, params):
    """Do some checks of the `obj` API against `params`

        The metaclass sets up a ``test_something`` function that runs these
        checks on each (
        """
    assert obj.var == params['var']
    assert obj.get_var() == params['var']