from functools import partial
from pytest import raises
from ..argument import Argument, to_arguments
from ..field import Field
from ..inputfield import InputField
from ..scalars import String
from ..structures import NonNull
def test_to_arguments_deprecated():
    args = {'unmounted_arg': String(required=False, deprecation_reason='deprecated')}
    my_args = to_arguments(args)
    assert my_args == {'unmounted_arg': Argument(String, required=False, deprecation_reason='deprecated')}