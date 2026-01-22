import threading
from typing import TYPE_CHECKING, Any, Dict, Optional
from ray.train._internal import session
from ray.train._internal.storage import StorageContext
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI(stability='stable')
class TrainContext:
    """Context for Ray training executions."""

    @_copy_doc(session.get_metadata)
    def get_metadata(self) -> Dict[str, Any]:
        return session.get_metadata()

    @_copy_doc(session.get_experiment_name)
    def get_experiment_name(self) -> str:
        return session.get_experiment_name()

    @_copy_doc(session.get_trial_name)
    def get_trial_name(self) -> str:
        return session.get_trial_name()

    @_copy_doc(session.get_trial_id)
    def get_trial_id(self) -> str:
        return session.get_trial_id()

    @_copy_doc(session.get_trial_resources)
    def get_trial_resources(self) -> 'PlacementGroupFactory':
        return session.get_trial_resources()

    @_copy_doc(session.get_trial_dir)
    def get_trial_dir(self) -> str:
        return session.get_trial_dir()

    @_copy_doc(session.get_world_size)
    def get_world_size(self) -> int:
        return session.get_world_size()

    @_copy_doc(session.get_world_rank)
    def get_world_rank(self) -> int:
        return session.get_world_rank()

    @_copy_doc(session.get_local_rank)
    def get_local_rank(self) -> int:
        return session.get_local_rank()

    @_copy_doc(session.get_local_world_size)
    def get_local_world_size(self) -> int:
        return session.get_local_world_size()

    @_copy_doc(session.get_node_rank)
    def get_node_rank(self) -> int:
        return session.get_node_rank()

    @DeveloperAPI
    @_copy_doc(session.get_storage)
    def get_storage(self) -> StorageContext:
        return session.get_storage()