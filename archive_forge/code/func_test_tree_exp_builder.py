import datetime
import pytest
import pyarrow as pa
@pytest.mark.gandiva
def test_tree_exp_builder():
    import pyarrow.gandiva as gandiva
    builder = gandiva.TreeExprBuilder()
    field_a = pa.field('a', pa.int32())
    field_b = pa.field('b', pa.int32())
    schema = pa.schema([field_a, field_b])
    field_result = pa.field('res', pa.int32())
    node_a = builder.make_field(field_a)
    node_b = builder.make_field(field_b)
    assert node_a.return_type() == field_a.type
    condition = builder.make_function('greater_than', [node_a, node_b], pa.bool_())
    if_node = builder.make_if(condition, node_a, node_b, pa.int32())
    expr = builder.make_expression(if_node, field_result)
    assert expr.result().type == pa.int32()
    config = gandiva.Configuration(dump_ir=True)
    projector = gandiva.make_projector(schema, [expr], pa.default_memory_pool(), 'NONE', config)
    assert projector.llvm_ir.find('@expr_') != -1
    a = pa.array([10, 12, -20, 5], type=pa.int32())
    b = pa.array([5, 15, 15, 17], type=pa.int32())
    e = pa.array([10, 15, 15, 17], type=pa.int32())
    input_batch = pa.RecordBatch.from_arrays([a, b], names=['a', 'b'])
    r, = projector.evaluate(input_batch)
    assert r.equals(e)