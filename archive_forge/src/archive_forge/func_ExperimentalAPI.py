from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def ExperimentalAPI(obj):
    """Decorator for documenting experimental APIs.

    Experimental APIs are classes and methods that are in development and may
    change at any time in their development process. You should not expect
    these APIs to be stable until their tag is changed to `DeveloperAPI` or
    `PublicAPI`.

    Subclasses that inherit from a ``@ExperimentalAPI`` base class can be
    assumed experimental as well.

    .. testcode::
        :skipif: True

        from ray.rllib.policy import Policy
        class TorchPolicy(Policy):
            ...
            # Indicates that the `TorchPolicy.loss` method is a new and
            # experimental API and may change frequently in future
            # releases.
            @ExperimentalAPI
            def loss(self, model, action_dist, train_batch):
                ...
    """
    _mark_annotated(obj)
    return obj