from mlflow.entities.view_type import ViewType
from mlflow.exceptions import MlflowException
@classmethod
def view_type_to_stages(cls, view_type=ViewType.ALL):
    stages = []
    if view_type == ViewType.ACTIVE_ONLY or view_type == ViewType.ALL:
        stages.append(cls.ACTIVE)
    if view_type == ViewType.DELETED_ONLY or view_type == ViewType.ALL:
        stages.append(cls.DELETED)
    return stages