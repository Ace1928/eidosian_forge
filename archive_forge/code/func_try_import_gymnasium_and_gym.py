import gymnasium as gym
from typing import Optional
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def try_import_gymnasium_and_gym():
    try:
        import gymnasium as gym
    except (ImportError, ModuleNotFoundError):
        raise ImportError('The `gymnasium` package seems to be not installed! As of Ray 2.2, it is required for RLlib. Try running `pip install gymnasium` from the command line to fix this problem.')
    old_gym = None
    try:
        import gym as old_gym
    except (ImportError, ModuleNotFoundError):
        pass
    return (gym, old_gym)