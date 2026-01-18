from functools import partial
from pytest import raises
from ..argument import Argument, to_arguments
from ..field import Field
from ..inputfield import InputField
from ..scalars import String
from ..structures import NonNull
def test_to_arguments_raises_if_field():
    args = {'arg_string': Field(String)}
    with raises(ValueError) as exc_info:
        to_arguments(args)
    assert str(exc_info.value) == 'Expected arg_string to be Argument, but received Field. Try using Argument(String).'