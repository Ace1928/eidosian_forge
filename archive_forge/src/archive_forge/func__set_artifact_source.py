from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
def _set_artifact_source(self, artifact: 'Artifact', name: Optional[str]=None) -> None:
    assert self._artifact_source is None, 'Cannot update artifact_source. Existing source: {}/{}'.format(self._artifact_source.artifact, self._artifact_source.name)
    self._artifact_source = _WBValueArtifactSource(artifact, name)