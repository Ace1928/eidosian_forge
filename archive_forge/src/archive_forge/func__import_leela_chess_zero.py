import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_leela_chess_zero():
    import ray.rllib.algorithms.leela_chess_zero as lc0
    return (lc0.LeelaChessZero, lc0.LeelaChessZero.get_default_config())