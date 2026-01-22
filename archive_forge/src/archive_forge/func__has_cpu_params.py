from sklearn.base import BaseEstimator
def _has_cpu_params(estimator: BaseEstimator) -> bool:
    """Returns True if estimator has any CPU-related params."""
    return any((any((param.endswith(cpu_param_name) for cpu_param_name in SKLEARN_CPU_PARAM_NAMES)) for param in estimator.get_params(deep=True)))