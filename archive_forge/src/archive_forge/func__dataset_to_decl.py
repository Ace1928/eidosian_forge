from pyarrow.lib import Table
from pyarrow.compute import Expression, field
def _dataset_to_decl(dataset, use_threads=True):
    decl = Declaration('scan', ScanNodeOptions(dataset, use_threads=use_threads))
    projections = [field(f) for f in dataset.schema.names]
    decl = Declaration.from_sequence([decl, Declaration('project', ProjectNodeOptions(projections))])
    filter_expr = dataset._scan_options.get('filter')
    if filter_expr is not None:
        decl = Declaration.from_sequence([decl, Declaration('filter', FilterNodeOptions(filter_expr))])
    return decl