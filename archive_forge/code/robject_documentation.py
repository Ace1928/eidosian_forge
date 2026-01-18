import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion

        R class for the object, stored as an R string vector.

        When setting the rclass, the new value will be:

        - wrapped in a Python tuple if a string (the R class
          is a vector of strings, and this is made for convenience)
        - wrapped in a StrSexpVector

        Note that when setting the class R may make a copy of
        the whole object (R is mostly a functional language).
        If this must be avoided, and if the number of parent
        classes before and after the change are compatible,
        the class name can be changed in-place by replacing
        vector elements.
        