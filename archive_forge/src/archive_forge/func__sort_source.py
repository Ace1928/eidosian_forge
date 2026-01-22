from pyarrow.lib import Table
from pyarrow.compute import Expression, field
def _sort_source(table_or_dataset, sort_keys, output_type=Table, **kwargs):
    if isinstance(table_or_dataset, ds.Dataset):
        data_source = _dataset_to_decl(table_or_dataset, use_threads=True)
    else:
        data_source = Declaration('table_source', TableSourceNodeOptions(table_or_dataset))
    order_by = Declaration('order_by', OrderByNodeOptions(sort_keys, **kwargs))
    decl = Declaration.from_sequence([data_source, order_by])
    result_table = decl.to_table(use_threads=True)
    if output_type == Table:
        return result_table
    elif output_type == ds.InMemoryDataset:
        return ds.InMemoryDataset(result_table)
    else:
        raise TypeError('Unsupported output type')