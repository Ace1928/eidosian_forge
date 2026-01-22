import inspect
import textwrap
class DeferredImportError(ImportError):
    """This exception is raised when something attempts to access a module
    that was imported by :py:func:`.attempt_import`, but the module
    import failed.

    """
    pass