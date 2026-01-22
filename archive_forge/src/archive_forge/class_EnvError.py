from ray.rllib.utils.annotations import PublicAPI
@PublicAPI
class EnvError(Exception):
    """Error if we encounter an error during RL environment validation."""
    pass