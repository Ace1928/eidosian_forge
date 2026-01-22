import warnings
from twisted.trial.unittest import TestCase
class METHOD(Names):
    """
            A container for some named constants to use in unit tests for
            L{Names}.
            """
    GET = NamedConstant()
    PUT = NamedConstant()
    POST = NamedConstant()
    DELETE = NamedConstant()
    extra = object()