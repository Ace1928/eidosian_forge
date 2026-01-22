import atexit
import pkg_resources
def GetResourceAsFile(name):
    """Get a resource as a file-like object; see _Call."""
    return _Call(pkg_resources.resource_stream, name)