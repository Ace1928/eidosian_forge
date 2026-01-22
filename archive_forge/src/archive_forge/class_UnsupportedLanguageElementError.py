class UnsupportedLanguageElementError(PyCTError, NotImplementedError):
    """Raised for code patterns that AutoGraph does not support."""