import inspect
import logging
import types
from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING
import ray
from ray.tune.execution.placement_groups import (
from ray.air.config import ScalingConfig
from ray.tune.registry import _ParameterRegistry
from ray.tune.utils import _detect_checkpoint_function
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def with_resources(trainable: Union[Type['Trainable'], Callable], resources: Union[Dict[str, float], PlacementGroupFactory, ScalingConfig, Callable[[dict], PlacementGroupFactory]]):
    """Wrapper for trainables to specify resource requests.

    This wrapper allows specification of resource requirements for a specific
    trainable. It will override potential existing resource requests (use
    with caution!).

    The main use case is to request resources for function trainables when used
    with the Tuner() API.

    Class trainables should usually just implement the ``default_resource_request()``
    method.

    Args:
        trainable: Trainable to wrap.
        resources: Resource dict, placement group factory, ``ScalingConfig``
            or callable that takes in a config dict and returns a placement
            group factory.

    Example:

    .. code-block:: python

        from ray import tune
        from ray.tune.tuner import Tuner

        def train_fn(config):
            return len(ray.get_gpu_ids())  # Returns 2

        tuner = Tuner(
            tune.with_resources(train_fn, resources={"gpu": 2}),
            # ...
        )
        results = tuner.fit()

    """
    from ray.tune.trainable import Trainable
    if not callable(trainable) or (inspect.isclass(trainable) and (not issubclass(trainable, Trainable))):
        raise ValueError(f'`tune.with_resources() only works with function trainables or classes that inherit from `tune.Trainable()`. Got type: {type(trainable)}.')
    if isinstance(resources, PlacementGroupFactory):
        pgf = resources
    elif isinstance(resources, ScalingConfig):
        pgf = resources.as_placement_group_factory()
    elif isinstance(resources, dict):
        pgf = resource_dict_to_pg_factory(resources)
    elif callable(resources):
        pgf = resources
    else:
        raise ValueError(f'Invalid resource type for `with_resources()`: {type(resources)}')
    if not inspect.isclass(trainable):
        if isinstance(trainable, types.MethodType):
            if _detect_checkpoint_function(trainable, partial=True):
                from ray.tune.trainable.function_trainable import _CHECKPOINT_DIR_ARG_DEPRECATION_MSG
                raise DeprecationWarning(_CHECKPOINT_DIR_ARG_DEPRECATION_MSG)

            def _trainable(config):
                return trainable(config)
            _trainable._resources = pgf
            return _trainable
        try:
            trainable._resources = pgf
        except AttributeError as e:
            raise RuntimeError('Could not use `tune.with_resources()` on the supplied trainable. Wrap your trainable in a regular function before passing it to Ray Tune.') from e
    else:

        class ResourceTrainable(trainable):

            @classmethod
            def default_resource_request(cls, config: Dict[str, Any]) -> Optional[PlacementGroupFactory]:
                if not isinstance(pgf, PlacementGroupFactory) and callable(pgf):
                    return pgf(config)
                return pgf
        ResourceTrainable.__name__ = trainable.__name__
        trainable = ResourceTrainable
    return trainable