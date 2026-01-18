import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@pytest.mark.parametrize('arr,schema_accessor,bad_type,good_type', [(pa.array(['a', 'b', 'c']), lambda x: x.type, pa.int32(), pa.string()), (pa.record_batch([pa.array(['a', 'b', 'c'])], names=['x']), lambda x: x.schema, pa.schema({'x': pa.int32()}), pa.schema({'x': pa.string()}))], ids=['array', 'record_batch'])
def test_roundtrip_array_capsule(arr, schema_accessor, bad_type, good_type):
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    import_array = type(arr)._import_from_c_capsule
    schema_capsule, capsule = arr.__arrow_c_array__()
    assert PyCapsule_IsValid(schema_capsule, b'arrow_schema') == 1
    assert PyCapsule_IsValid(capsule, b'arrow_array') == 1
    arr_out = import_array(schema_capsule, capsule)
    assert arr_out.equals(arr)
    assert pa.total_allocated_bytes() > old_allocated
    del arr_out
    assert pa.total_allocated_bytes() == old_allocated
    capsule = arr.__arrow_c_array__()
    assert pa.total_allocated_bytes() > old_allocated
    del capsule
    assert pa.total_allocated_bytes() == old_allocated
    with pytest.raises(ValueError, match='Could not cast.* string to requested .* int32'):
        arr.__arrow_c_array__(bad_type.__arrow_c_schema__())
    schema_capsule, array_capsule = arr.__arrow_c_array__(good_type.__arrow_c_schema__())
    arr_out = import_array(schema_capsule, array_capsule)
    assert schema_accessor(arr_out) == good_type