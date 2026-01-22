def _is_column_extension_type(ca: 'pyarrow.ChunkedArray') -> bool:
    """Whether the provided Arrow Table column is an extension array, using an Arrow
    extension type.
    """
    return isinstance(ca.type, pyarrow.ExtensionType)