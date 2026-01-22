from typing import TYPE_CHECKING, Optional
from wandb import errors
class ArtifactStatusError(AttributeError):
    """Raised when an artifact is in an invalid state for the requested operation."""

    def __init__(self, artifact: Optional['Artifact']=None, attr: Optional[str]=None, msg: str='Artifact is in an invalid state for the requested operation.'):
        object_name = artifact.__class__.__name__ if artifact else 'Artifact'
        method_id = f'{object_name}.{attr}' if attr else object_name
        super().__init__(msg.format(artifact=artifact, attr=attr, method_id=method_id))
        self.obj = artifact
        self.name = attr or ''