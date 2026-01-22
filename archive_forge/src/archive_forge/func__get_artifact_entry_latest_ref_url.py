from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
def _get_artifact_entry_latest_ref_url(self) -> Optional[str]:
    if self._artifact_target and self._artifact_target.name and (self._artifact_target.artifact._client_id is not None) and self._artifact_target.artifact._final and _server_accepts_client_ids():
        return 'wandb-client-artifact://{}:latest/{}'.format(self._artifact_target.artifact._sequence_client_id, type(self).with_suffix(self._artifact_target.name))
    elif self._artifact_target and self._artifact_target.name and self._artifact_target.artifact._is_draft_save_started() and (not util._is_offline()) and (not _server_accepts_client_ids()):
        self._artifact_target.artifact.wait()
        ref_entry = self._artifact_target.artifact.get_entry(type(self).with_suffix(self._artifact_target.name))
        return str(ref_entry.ref_url())
    return None