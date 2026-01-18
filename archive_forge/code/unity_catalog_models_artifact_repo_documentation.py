from mlflow.environment_variables import MLFLOW_UNITY_CATALOG_PRESIGNED_URLS_ENABLED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.store.artifact.utils.models import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
from mlflow.utils.uri import (

        Get underlying ArtifactRepository instance for model version blob
        storage
        