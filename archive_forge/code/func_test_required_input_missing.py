from ...types import ObjectType, Schema, String, NonNull
def test_required_input_missing():
    """
    Test that a required argument raised an error if not provided.
    """
    result = schema.execute('{ hello }')
    assert result.errors
    assert len(result.errors) == 1
    assert result.errors[0].message == "Field 'hello' argument 'input' of type 'String!' is required, but it was not provided."