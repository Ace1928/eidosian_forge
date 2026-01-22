import inspect
import sys
from zunclient.i18n import _
class NoUniqueMatch(ClientException):
    """Multiple entities found instead of one."""
    pass