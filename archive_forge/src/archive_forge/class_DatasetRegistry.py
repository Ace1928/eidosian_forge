import inspect
import warnings
from contextlib import suppress
from typing import Dict, Optional
import entrypoints
import mlflow.data
from mlflow.data.dataset import Dataset
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
class DatasetRegistry:

    def __init__(self):
        self.constructors = {}

    def register_constructor(self, constructor_fn: callable, constructor_name: Optional[str]=None) -> str:
        """Registers a dataset constructor.

        Args:
            constructor_fn: A function that accepts at least the following
                inputs and returns an instance of a subclass of
                :py:class:`mlflow.data.dataset.Dataset`:

                - name: Optional. A string dataset name
                - digest: Optional. A string dataset digest.

            constructor_name: The name of the constructor, e.g.
                "from_spark". The name must begin with the
                string "from_" or "load_". If unspecified, the `__name__`
                attribute of the `constructor_fn` is used instead and must
                begin with the string "from_" or "load_".

        Returns:
            The name of the registered constructor, e.g. "from_pandas" or "load_delta".
        """
        if constructor_name is None:
            constructor_name = constructor_fn.__name__
        DatasetRegistry._validate_constructor(constructor_fn, constructor_name)
        self.constructors[constructor_name] = constructor_fn
        return constructor_name

    def register_entrypoints(self):
        """
        Registers dataset sources defined as Python entrypoints. For reference, see
        https://mlflow.org/docs/latest/plugins.html#defining-a-plugin.
        """
        for entrypoint in entrypoints.get_group_all('mlflow.dataset_constructor'):
            try:
                self.register_constructor(constructor_fn=entrypoint.load(), constructor_name=entrypoint.name)
            except Exception as exc:
                warnings.warn(f'Failure attempting to register dataset constructor "{entrypoint.name}": {exc}.', stacklevel=2)

    @staticmethod
    def _validate_constructor(constructor_fn: callable, constructor_name: str):
        if not constructor_name.startswith('load_') and (not constructor_name.startswith('from_')):
            raise MlflowException(f"Invalid dataset constructor name: {constructor_name}. Constructor name must start with 'load_' or 'from_'.", INVALID_PARAMETER_VALUE)
        signature = inspect.signature(constructor_fn)
        parameters = signature.parameters
        for expected_kwarg in ['name', 'digest']:
            if expected_kwarg not in parameters or parameters[expected_kwarg].kind not in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
                raise MlflowException(f"Invalid dataset constructor function: {constructor_fn.__name__}. Function must define an optional parameter named '{expected_kwarg}'.", INVALID_PARAMETER_VALUE)
        if not issubclass(signature.return_annotation, Dataset):
            raise MlflowException(f'Invalid dataset constructor function: {constructor_fn.__name__}. Function must have a return type annotation that is a subclass of :py:class:`mlflow.data.dataset.Dataset`.', INVALID_PARAMETER_VALUE)