import weakref
import importlib
from numba import _dynfunc
def _rebuild_env(modname, consts, env_name):
    env = lookup_environment(env_name)
    if env is not None:
        return env
    mod = importlib.import_module(modname)
    env = Environment(mod.__dict__)
    env.consts[:] = consts
    env.env_name = env_name
    Environment._memo[env_name] = env
    return env