import logging
import warnings
import entrypoints
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.default_experiment.databricks_notebook_experiment_provider import (
class DefaultExperimentProviderRegistry:
    """Registry for default experiment provider implementations

    This class allows the registration of default experiment providers, which are used to provide
    MLflow Experiment IDs based on the current context where the MLflow client is running when
    the user has not explicitly set an experiment. Implementations declared though the entrypoints
    `mlflow.default_experiment_provider` group can be automatically registered through the
    `register_entrypoints` method.
    """

    def __init__(self):
        self._registry = []

    def register(self, default_experiment_provider_cls):
        self._registry.append(default_experiment_provider_cls())

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all('mlflow.default_experiment_provider'):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn('Failure attempting to register default experiment' + f'context provider "{entrypoint.name}": {exc}', stacklevel=2)

    def __iter__(self):
        return iter(self._registry)