from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
def test_subclassed_mutation_input():
    Input = OtherMutation.Input
    assert issubclass(Input, InputObjectType)
    fields = Input._meta.fields
    assert list(fields) == ['shared', 'additional_field', 'client_mutation_id']
    assert isinstance(fields['shared'], InputField)
    assert fields['shared'].type == String
    assert isinstance(fields['additional_field'], InputField)
    assert fields['additional_field'].type == String
    assert isinstance(fields['client_mutation_id'], InputField)
    assert fields['client_mutation_id'].type == String