import gymnasium as gym
from typing import Type
Wrapper for gym.Envs to have their action space transformed.

    Args:
        env_name_or_creator (Union[str, Callable[]]: String specifier or
            env_maker function.

    Returns:
        New transformed_action_space_env function that returns an environment
        wrapped by the ActionTransform wrapper. The constructor takes a
        config dict with `_low` and `_high` keys specifying the new action
        range (default -1.0 to 1.0). The reset of the config dict will be
        passed on to the underlying/wrapped env's constructor.

    .. testcode::
        :skipif: True

        # By gym string:
        pendulum_300_to_500_cls = transform_action_space("Pendulum-v1")
        # Create a transformed pendulum env.
        pendulum_300_to_500 = pendulum_300_to_500_cls({"_low": -15.0})
        pendulum_300_to_500.action_space

    .. testoutput::

        gym.spaces.Box(-15.0, 1.0, (1, ), "float32")
    