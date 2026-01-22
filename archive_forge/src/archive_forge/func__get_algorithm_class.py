import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _get_algorithm_class(alg: str) -> type:
    if alg in ALGORITHMS:
        return ALGORITHMS[alg]()[0]
    elif alg == 'script':
        from ray.tune import script_runner
        return script_runner.ScriptRunner
    elif alg == '__fake':
        from ray.rllib.algorithms.mock import _MockTrainer
        return _MockTrainer
    elif alg == '__sigmoid_fake_data':
        from ray.rllib.algorithms.mock import _SigmoidFakeData
        return _SigmoidFakeData
    elif alg == '__parameter_tuning':
        from ray.rllib.algorithms.mock import _ParameterTuningTrainer
        return _ParameterTuningTrainer
    else:
        raise Exception('Unknown algorithm {}.'.format(alg))