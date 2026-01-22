import pytest
import sys
import textwrap
import rpy2.robjects as robjects
import rpy2.robjects.methods as methods
class MLE(object, metaclass=robjects.methods.RS4Auto_Type):
    __rname__ = 'mle'