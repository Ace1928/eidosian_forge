import logging
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import append_to_uri_path
def _get_flavor_backend_for_local_model(model=None, build_docker=True, **kwargs):
    from mlflow import pyfunc
    from mlflow.pyfunc.backend import PyFuncBackend
    from mlflow.rfunc.backend import RFuncBackend
    if not model:
        return (pyfunc.FLAVOR_NAME, PyFuncBackend({}, **kwargs))
    _flavor_backends = {pyfunc.FLAVOR_NAME: PyFuncBackend, 'crate': RFuncBackend}
    for flavor_name, flavor_config in model.flavors.items():
        if flavor_name in _flavor_backends:
            backend = _flavor_backends[flavor_name](flavor_config, **kwargs)
            if build_docker and backend.can_build_image() or backend.can_score_model():
                return (flavor_name, backend)
    return (None, None)