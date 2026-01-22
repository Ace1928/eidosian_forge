import sys
from .dependencies import random
class FlagType(type):
    """Metaclass to simplify the repr(type) and str(type)

    This metaclass redefines the ``str()`` and ``repr()`` of resulting
    classes.  The str() of the class returns only the class' ``__name__``,
    whereas the repr() returns either the qualified class name
    (``__qualname__``) if Sphinx has been imported, or else the
    fully-qualified class name (``__module__ + '.' + __qualname__``).

    This is useful for defining "flag types" that are default arguments
    in functions so that the Sphinx-generated documentation is "cleaner"

    """
    if 'sphinx' in sys.modules or 'Sphinx' in sys.modules:

        def __repr__(cls):
            return cls.__qualname__
    else:

        def __repr__(cls):
            return cls.__module__ + '.' + cls.__qualname__

    def __str__(cls):
        return cls.__name__