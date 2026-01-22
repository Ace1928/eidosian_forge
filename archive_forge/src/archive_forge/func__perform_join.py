from pyarrow.lib import Table
from pyarrow.compute import Expression, field
def _perform_join(join_type, left_operand, left_keys, right_operand, right_keys, left_suffix=None, right_suffix=None, use_threads=True, coalesce_keys=False, output_type=Table):
    """
    Perform join of two tables or datasets.

    The result will be an output table with the result of the join operation

    Parameters
    ----------
    join_type : str
        One of supported join types.
    left_operand : Table or Dataset
        The left operand for the join operation.
    left_keys : str or list[str]
        The left key (or keys) on which the join operation should be performed.
    right_operand : Table or Dataset
        The right operand for the join operation.
    right_keys : str or list[str]
        The right key (or keys) on which the join operation should be performed.
    left_suffix : str, default None
        Which suffix to add to left column names. This prevents confusion
        when the columns in left and right operands have colliding names.
    right_suffix : str, default None
        Which suffix to add to the right column names. This prevents confusion
        when the columns in left and right operands have colliding names.
    use_threads : bool, default True
        Whether to use multithreading or not.
    coalesce_keys : bool, default False
        If the duplicated keys should be omitted from one of the sides
        in the join result.
    output_type: Table or InMemoryDataset
        The output type for the exec plan result.

    Returns
    -------
    result_table : Table or InMemoryDataset
    """
    if not isinstance(left_operand, (Table, ds.Dataset)):
        raise TypeError(f'Expected Table or Dataset, got {type(left_operand)}')
    if not isinstance(right_operand, (Table, ds.Dataset)):
        raise TypeError(f'Expected Table or Dataset, got {type(right_operand)}')
    left_keys_order = {}
    if not isinstance(left_keys, (tuple, list)):
        left_keys = [left_keys]
    for idx, key in enumerate(left_keys):
        left_keys_order[key] = idx
    right_keys_order = {}
    if not isinstance(right_keys, (list, tuple)):
        right_keys = [right_keys]
    for idx, key in enumerate(right_keys):
        right_keys_order[key] = idx
    left_columns = left_operand.schema.names
    right_columns = right_operand.schema.names
    if join_type == 'left semi' or join_type == 'left anti':
        right_columns = []
    elif join_type == 'right semi' or join_type == 'right anti':
        left_columns = []
    elif join_type == 'inner' or join_type == 'left outer':
        right_columns = [col for col in right_columns if col not in right_keys_order]
    elif join_type == 'right outer':
        left_columns = [col for col in left_columns if col not in left_keys_order]
    left_column_keys_indices = {}
    for idx, colname in enumerate(left_columns):
        if colname in left_keys:
            left_column_keys_indices[colname] = idx
    right_column_keys_indices = {}
    for idx, colname in enumerate(right_columns):
        if colname in right_keys:
            right_column_keys_indices[colname] = idx
    if isinstance(left_operand, ds.Dataset):
        left_source = _dataset_to_decl(left_operand, use_threads=use_threads)
    else:
        left_source = Declaration('table_source', TableSourceNodeOptions(left_operand))
    if isinstance(right_operand, ds.Dataset):
        right_source = _dataset_to_decl(right_operand, use_threads=use_threads)
    else:
        right_source = Declaration('table_source', TableSourceNodeOptions(right_operand))
    if coalesce_keys:
        join_opts = HashJoinNodeOptions(join_type, left_keys, right_keys, left_columns, right_columns, output_suffix_for_left=left_suffix or '', output_suffix_for_right=right_suffix or '')
    else:
        join_opts = HashJoinNodeOptions(join_type, left_keys, right_keys, output_suffix_for_left=left_suffix or '', output_suffix_for_right=right_suffix or '')
    decl = Declaration('hashjoin', options=join_opts, inputs=[left_source, right_source])
    if coalesce_keys and join_type == 'full outer':
        left_columns_set = set(left_columns)
        right_columns_set = set(right_columns)
        right_operand_index = len(left_columns)
        projected_col_names = []
        projections = []
        for idx, col in enumerate(left_columns + right_columns):
            if idx < len(left_columns) and col in left_column_keys_indices:
                projected_col_names.append(col)
                right_key_index = right_column_keys_indices[right_keys[left_keys_order[col]]]
                projections.append(Expression._call('coalesce', [Expression._field(idx), Expression._field(right_operand_index + right_key_index)]))
            elif idx >= right_operand_index and col in right_column_keys_indices:
                continue
            else:
                if left_suffix and idx < right_operand_index and (col in right_columns_set):
                    col += left_suffix
                if right_suffix and idx >= right_operand_index and (col in left_columns_set):
                    col += right_suffix
                projected_col_names.append(col)
                projections.append(Expression._field(idx))
        projection = Declaration('project', ProjectNodeOptions(projections, projected_col_names))
        decl = Declaration.from_sequence([decl, projection])
    result_table = decl.to_table(use_threads=use_threads)
    if output_type == Table:
        return result_table
    elif output_type == ds.InMemoryDataset:
        return ds.InMemoryDataset(result_table)
    else:
        raise TypeError('Unsupported output type')