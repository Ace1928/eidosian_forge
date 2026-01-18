import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product

        Check that the correct error messages are raised while executing:
          with method(*args):
              func()
        *errors* should be a list of 4 regex that match the error when:
          1) longMessage = False and no msg passed;
          2) longMessage = False and msg passed;
          3) longMessage = True and no msg passed;
          4) longMessage = True and msg passed;
        