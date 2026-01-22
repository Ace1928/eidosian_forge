from enum import Enum
class ExportError(Exception):
    """
    This type of exception is raised for errors that are directly caused by the user
    code. In general, user errors happen during model authoring, tracing, using our public
    facing APIs, and writing graph passes.
    """

    def __init__(self, error_code: ExportErrorType, message: str) -> None:
        prefix = f'[{error_code}]: '
        super().__init__(prefix + message)