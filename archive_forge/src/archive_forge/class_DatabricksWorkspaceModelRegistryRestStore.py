from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.rest_store import RestStore
class DatabricksWorkspaceModelRegistryRestStore(RestStore):

    def set_registered_model_alias(self, name, alias, version):
        _raise_unsupported_method(method='set_registered_model_alias')

    def delete_registered_model_alias(self, name, alias):
        _raise_unsupported_method(method='delete_registered_model_alias')

    def get_model_version_by_alias(self, name, alias):
        _raise_unsupported_method(method='get_model_version_by_alias', message="If attempting to load a model version by alias via a URI of the form 'models:/model_name@alias_name', configure the MLflow client to target Unity Catalog and try again.")

    def _await_model_version_creation(self, mv, await_creation_for):
        uc_hint = ' For faster model version creation, use Models in Unity Catalog (https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html).'
        self._await_model_version_creation_impl(mv, await_creation_for, hint=uc_hint)