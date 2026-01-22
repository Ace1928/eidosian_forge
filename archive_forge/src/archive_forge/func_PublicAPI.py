from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def PublicAPI(obj):
    """Decorator for documenting public APIs.

    Public APIs are classes and methods exposed to end users of RLlib. You
    can expect these APIs to remain stable across RLlib releases.

    Subclasses that inherit from a ``@PublicAPI`` base class can be
    assumed part of the RLlib public API as well (e.g., all Algorithm classes
    are in public API because Algorithm is ``@PublicAPI``).

    In addition, you can assume all algo configurations are part of their
    public API as well.

    .. testcode::
        :skipif: True

        # Indicates that the `Algorithm` class is exposed to end users
        # of RLlib and will remain stable across RLlib releases.
        from ray import tune
        @PublicAPI
        class Algorithm(tune.Trainable):
            ...
    """
    _mark_annotated(obj)
    return obj