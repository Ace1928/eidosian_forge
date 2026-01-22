import pytest
from nipype.utils.functions import getsource, create_function_from_source
def _func1(x):
    return x ** 3