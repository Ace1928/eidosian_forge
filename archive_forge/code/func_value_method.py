from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def value_method(self):
    return 'value_method'