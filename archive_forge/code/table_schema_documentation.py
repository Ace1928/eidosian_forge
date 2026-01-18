from pandas.core.common import (

    Create a Table schema from ``data``.

    Parameters
    ----------
    data : Series, DataFrame
    index : bool, default True
        Whether to include ``data.index`` in the schema.
    primary_key : bool or None, default True
        column names to designate as the primary key.
        The default `None` will set `'primaryKey'` to the index
        level or levels if the index is unique.
    version : bool, default True
        Whether to include a field `pandas_version` with the version
        of pandas that generated the schema.

    Returns
    -------
    schema : dict

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {'A': [1, 2, 3],
    ...      'B': ['a', 'b', 'c'],
    ...      'C': pd.date_range('2016-01-01', freq='d', periods=3),
    ...     }, index=pd.Index(range(3), name='idx'))
    >>> build_table_schema(df)
    {'fields': [{'name': 'idx', 'type': 'integer'},
    {'name': 'A', 'type': 'integer'},
    {'name': 'B', 'type': 'string'},
    {'name': 'C', 'type': 'datetime'}],
    'pandas_version': '0.20.0',
    'primaryKey': ['idx']}

    Notes
    -----
    See `_as_json_table_type` for conversion types.
    Timedeltas as converted to ISO8601 duration format with
    9 decimal places after the secnods field for nanosecond precision.

    Categoricals are converted to the `any` dtype, and use the `enum` field
    constraint to list the allowed values. The `ordered` attribute is included
    in an `ordered` field.
    