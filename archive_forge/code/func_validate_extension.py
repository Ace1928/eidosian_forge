import importlib
import time
import warnings
def validate_extension(name):
    """Raises an exception is the extension is missing a needed
    hook or metadata field.
    An extension is valid if:
    1) name is an importable Python package.
    1) the package has a _jupyter_server_extension_points function
    2) each extension path has a _load_jupyter_server_extension function

    If this works, nothing should happen.
    """
    from .manager import ExtensionPackage
    return ExtensionPackage(name=name)