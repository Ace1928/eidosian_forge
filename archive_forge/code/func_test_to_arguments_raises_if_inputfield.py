from functools import partial
from pytest import raises
from ..argument import Argument, to_arguments
from ..field import Field
from ..inputfield import InputField
from ..scalars import String
from ..structures import NonNull
def test_to_arguments_raises_if_inputfield():
    args = {'arg_string': InputField(String)}
    with raises(ValueError) as exc_info:
        to_arguments(args)
    assert str(exc_info.value) == 'Expected arg_string to be Argument, but received InputField. Try using Argument(String).'