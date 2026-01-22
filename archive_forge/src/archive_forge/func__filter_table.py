from pyarrow.lib import Table
from pyarrow.compute import Expression, field
def _filter_table(table, expression):
    """Filter rows of a table based on the provided expression.

    The result will be an output table with only the rows matching
    the provided expression.

    Parameters
    ----------
    table : Table or Dataset
        Table or Dataset that should be filtered.
    expression : Expression
        The expression on which rows should be filtered.

    Returns
    -------
    Table
    """
    decl = Declaration.from_sequence([Declaration('table_source', options=TableSourceNodeOptions(table)), Declaration('filter', options=FilterNodeOptions(expression))])
    return decl.to_table(use_threads=True)