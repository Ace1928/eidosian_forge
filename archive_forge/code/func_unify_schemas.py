from typing import TYPE_CHECKING, List, Union
def unify_schemas(schemas: List['pyarrow.Schema']) -> 'pyarrow.Schema':
    """Version of `pyarrow.unify_schemas()` which also handles checks for
    variable-shaped tensors in the given schemas."""
    import pyarrow as pa
    from ray.air.util.tensor_extensions.arrow import ArrowTensorType, ArrowVariableShapedTensorType
    schemas_to_unify = []
    schema_field_overrides = {}
    cols_with_null_list = set()
    for schema in schemas:
        for col_name in schema.names:
            col_type = schema.field(col_name).type
            if pa.types.is_list(col_type) and pa.types.is_null(col_type.value_type):
                cols_with_null_list.add(col_name)
    if any((isinstance(type_, pyarrow.ExtensionType) for type_ in schemas[0].types)):
        for col_field in schemas[0]:
            col_name, col_type = (col_field.name, col_field.type)
            tensor_array_types = [s.field(col_name).type for s in schemas if isinstance(s.field(col_name).type, pyarrow.ExtensionType)]
            if ArrowTensorType._need_variable_shaped_tensor_array(tensor_array_types):
                if isinstance(tensor_array_types[0], ArrowVariableShapedTensorType):
                    new_type = tensor_array_types[0]
                elif isinstance(tensor_array_types[0], ArrowTensorType):
                    new_type = ArrowVariableShapedTensorType(dtype=tensor_array_types[0].scalar_type, ndim=len(tensor_array_types[0].shape))
                else:
                    raise ValueError(f'Detected need for variable shaped tensor representation, but schema is not ArrayTensorType: {tensor_array_types[0]}')
                schema_field_overrides[col_name] = new_type
    if cols_with_null_list:
        for col_name in cols_with_null_list:
            for schema in schemas:
                col_type = schema.field(col_name).type
                if not pa.types.is_list(col_type) or not pa.types.is_null(col_type.value_type):
                    schema_field_overrides[col_name] = col_type
                    break
    if schema_field_overrides:
        for schema in schemas:
            for col_name, col_new_type in schema_field_overrides.items():
                var_shaped_col = schema.field(col_name).with_type(col_new_type)
                col_idx = schema.get_field_index(col_name)
                schema = schema.set(col_idx, var_shaped_col)
            schemas_to_unify.append(schema)
    else:
        schemas_to_unify = schemas
    return pyarrow.unify_schemas(schemas_to_unify)