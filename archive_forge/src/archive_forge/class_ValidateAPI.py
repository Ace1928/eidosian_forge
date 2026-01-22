import os
import pytest
class ValidateAPI(metaclass=validator2test):
    """A class to validate APIs

    Your job is twofold:

    * define an ``obj_params`` iterable, where the iterator returns (``obj``,
      ``params``) pairs.  ``obj`` is something that you want to validate against
      an API.  ``params`` is a mapping giving parameters for this object to test
      against.
    * define ``validate_xxx`` methods, that accept ``obj`` and
      ``params`` as arguments, and check ``obj`` against ``params``

    The metaclass finds each ``validate_xxx`` method and makes a new
    ``test_xxx`` method that calls ``validate_xxx`` for each (``obj``,
    ``params``) pair returned from ``obj_params``

    See :class:`TextValidateSomething` for an example
    """