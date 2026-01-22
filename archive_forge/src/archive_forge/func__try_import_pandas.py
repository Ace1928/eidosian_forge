def _try_import_pandas() -> bool:
    try:
        import pandas
        global _pandas
        _pandas = pandas
        return True
    except ImportError:
        return False