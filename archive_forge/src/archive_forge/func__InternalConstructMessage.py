def _InternalConstructMessage(full_name):
    """Constructs a nested message."""
    from google.protobuf import symbol_database
    return symbol_database.Default().GetSymbol(full_name)()