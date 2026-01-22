import dataclasses
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from ray._private.storage import _get_storage_uri
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.syncer import SyncConfig, Syncer, _BackgroundSyncer
from ray.train.constants import _get_defaults_results_dir
class StorageContext:
    """Shared context that holds all paths and storage utilities, passed along from
    the driver to workers.

    The properties of this context may not all be set at once, depending on where
    the context lives.
    For example, on the driver, the storage context is initialized, only knowing
    the experiment path. On the Trainable actor, the trial_dir_name is accessible.

    There are 2 types of paths:
    1. *_fs_path: A path on the `storage_filesystem`. This is a regular path
        which has been prefix-stripped by pyarrow.fs.FileSystem.from_uri and
        can be joined with `os.path.join`.
    2. *_local_path: The path on the local filesystem where results are saved to
       before persisting to storage.

    Example with storage_path="mock:///bucket/path?param=1":

        >>> from ray.train._internal.storage import StorageContext
        >>> import os
        >>> os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = "/tmp/ray_results"
        >>> storage = StorageContext(
        ...     storage_path="mock://netloc/bucket/path?param=1",
        ...     experiment_dir_name="exp_name",
        ... )
        >>> storage.storage_filesystem   # Auto-resolved  # doctest: +ELLIPSIS
        <pyarrow._fs._MockFileSystem object...
        >>> storage.experiment_fs_path
        'bucket/path/exp_name'
        >>> storage.experiment_local_path
        '/tmp/ray_results/exp_name'
        >>> storage.trial_dir_name = "trial_dir"
        >>> storage.trial_fs_path
        'bucket/path/exp_name/trial_dir'
        >>> storage.trial_local_path
        '/tmp/ray_results/exp_name/trial_dir'
        >>> storage.current_checkpoint_index = 1
        >>> storage.checkpoint_fs_path
        'bucket/path/exp_name/trial_dir/checkpoint_000001'

    Example with storage_path=None:

        >>> from ray.train._internal.storage import StorageContext
        >>> import os
        >>> os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = "/tmp/ray_results"
        >>> storage = StorageContext(
        ...     storage_path=None,
        ...     experiment_dir_name="exp_name",
        ... )
        >>> storage.storage_fs_path  # Auto-resolved
        '/tmp/ray_results'
        >>> storage.storage_local_path
        '/tmp/ray_results'
        >>> storage.experiment_local_path
        '/tmp/ray_results/exp_name'
        >>> storage.experiment_fs_path
        '/tmp/ray_results/exp_name'
        >>> storage.syncer is None
        True
        >>> storage.storage_filesystem   # Auto-resolved  # doctest: +ELLIPSIS
        <pyarrow._fs.LocalFileSystem object...

    Internal Usage Examples:
    - To copy files to the trial directory on the storage filesystem:

        pyarrow.fs.copy_files(
            local_dir,
            os.path.join(storage.trial_fs_path, "subdir"),
            destination_filesystem=storage.filesystem
        )
    """

    def __init__(self, storage_path: Optional[Union[str, os.PathLike]], experiment_dir_name: str, sync_config: Optional[SyncConfig]=None, storage_filesystem: Optional[pyarrow.fs.FileSystem]=None, trial_dir_name: Optional[str]=None, current_checkpoint_index: int=-1):
        self.custom_fs_provided = storage_filesystem is not None
        self.storage_local_path = _get_defaults_results_dir()
        ray_storage_uri: Optional[str] = _get_storage_uri()
        if ray_storage_uri and storage_path is None:
            logger.info(f'Using configured Ray Storage URI as the `storage_path`: {ray_storage_uri}')
        storage_path = storage_path or ray_storage_uri or self.storage_local_path
        self.experiment_dir_name = experiment_dir_name
        self.trial_dir_name = trial_dir_name
        self.current_checkpoint_index = current_checkpoint_index
        self.sync_config = dataclasses.replace(sync_config) if sync_config else SyncConfig()
        self.storage_filesystem, self.storage_fs_path = get_fs_and_path(storage_path, storage_filesystem)
        self.storage_fs_path = Path(self.storage_fs_path).as_posix()
        syncing_needed = self.custom_fs_provided or self.storage_fs_path != self.storage_local_path
        self.syncer: Optional[Syncer] = _FilesystemSyncer(storage_filesystem=self.storage_filesystem, sync_period=self.sync_config.sync_period, sync_timeout=self.sync_config.sync_timeout) if syncing_needed else None
        self._create_validation_file()
        self._check_validation_file()

    def __str__(self):
        return f"StorageContext<\n  storage_filesystem='{self.storage_filesystem.type_name}',\n  storage_fs_path='{self.storage_fs_path}',\n  storage_local_path='{self.storage_local_path}',\n  experiment_dir_name='{self.experiment_dir_name}',\n  trial_dir_name='{self.trial_dir_name}',\n  current_checkpoint_index={self.current_checkpoint_index},\n>"

    def _create_validation_file(self):
        """On the creation of a storage context, create a validation file at the
        storage path to verify that the storage path can be written to.
        This validation file is also used to check whether the storage path is
        accessible by all nodes in the cluster."""
        valid_file = os.path.join(self.experiment_fs_path, _VALIDATE_STORAGE_MARKER_FILENAME)
        self.storage_filesystem.create_dir(self.experiment_fs_path)
        with self.storage_filesystem.open_output_stream(valid_file):
            pass

    def _check_validation_file(self):
        """Checks that the validation file exists at the storage path."""
        valid_file = os.path.join(self.experiment_fs_path, _VALIDATE_STORAGE_MARKER_FILENAME)
        if not _exists_at_fs_path(fs=self.storage_filesystem, fs_path=valid_file):
            raise RuntimeError(f"Unable to set up cluster storage with the following settings:\n{self}\nCheck that all nodes in the cluster have read/write access to the configured storage path. `RunConfig(storage_path)` should be set to a cloud storage URI or a shared filesystem path accessible by all nodes in your cluster ('s3://bucket' or '/mnt/nfs'). A local path on the head node is not accessible by worker nodes. See: https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html")

    def _update_checkpoint_index(self, metrics: Dict):
        self.current_checkpoint_index += 1

    def persist_current_checkpoint(self, checkpoint: 'Checkpoint') -> 'Checkpoint':
        """Persists a given checkpoint to the current checkpoint path on the filesystem.

        "Current" is defined by the `current_checkpoint_index` attribute of the
        storage context.

        This method copies the checkpoint files to the storage location.
        It's up to the user to delete the original checkpoint files if desired.

        For example, the original directory is typically a local temp directory.

        Args:
            checkpoint: The checkpoint to persist to (fs, checkpoint_fs_path).

        Returns:
            Checkpoint: A Checkpoint pointing to the persisted checkpoint location.
        """
        from ray.train._checkpoint import Checkpoint
        logger.debug('Copying checkpoint files to storage path:\n({source_fs}, {source}) -> ({dest_fs}, {destination})'.format(source=checkpoint.path, destination=self.checkpoint_fs_path, source_fs=checkpoint.filesystem, dest_fs=self.storage_filesystem))
        self._check_validation_file()
        self.storage_filesystem.create_dir(self.checkpoint_fs_path)
        _pyarrow_fs_copy_files(source=checkpoint.path, destination=self.checkpoint_fs_path, source_filesystem=checkpoint.filesystem, destination_filesystem=self.storage_filesystem)
        persisted_checkpoint = Checkpoint(filesystem=self.storage_filesystem, path=self.checkpoint_fs_path)
        logger.info(f'Checkpoint successfully created at: {persisted_checkpoint}')
        return persisted_checkpoint

    def persist_artifacts(self, force: bool=False) -> None:
        """Persists all artifacts within `trial_local_dir` to storage.

        This method possibly launches a background task to sync the trial dir,
        depending on the `sync_period` + `sync_artifacts_on_checkpoint`
        settings of `SyncConfig`.

        `(local_fs, trial_local_path) -> (storage_filesystem, trial_fs_path)`

        Args:
            force: If True, wait for a previous sync to finish, launch a new one,
                and wait for that one to finish. By the end of a `force=True` call, the
                latest version of the trial artifacts will be persisted.
        """
        if not self.sync_config.sync_artifacts:
            return
        if not self.syncer:
            return
        if force:
            self.syncer.wait()
            self.syncer.sync_up(local_dir=self.trial_local_path, remote_dir=self.trial_fs_path)
            self.syncer.wait()
        else:
            self.syncer.sync_up_if_needed(local_dir=self.trial_local_path, remote_dir=self.trial_fs_path)

    @property
    def experiment_fs_path(self) -> str:
        """The path on the `storage_filesystem` to the experiment directory.

        NOTE: This does not have a URI prefix anymore, since it has been stripped
        by pyarrow.fs.FileSystem.from_uri already. The URI scheme information is
        kept in `storage_filesystem` instead.
        """
        return Path(self.storage_fs_path, self.experiment_dir_name).as_posix()

    @property
    def experiment_local_path(self) -> str:
        """The local filesystem path to the experiment directory.

        This local "cache" path refers to location where files are dumped before
        syncing them to the `storage_path` on the `storage_filesystem`.
        """
        return Path(self.storage_local_path, self.experiment_dir_name).as_posix()

    @property
    def trial_local_path(self) -> str:
        """The local filesystem path to the trial directory.

        Raises a ValueError if `trial_dir_name` is not set beforehand.
        """
        if self.trial_dir_name is None:
            raise RuntimeError('Should not access `trial_local_path` without setting `trial_dir_name`')
        return Path(self.experiment_local_path, self.trial_dir_name).as_posix()

    @property
    def trial_fs_path(self) -> str:
        """The trial directory path on the `storage_filesystem`.

        Raises a ValueError if `trial_dir_name` is not set beforehand.
        """
        if self.trial_dir_name is None:
            raise RuntimeError('Should not access `trial_fs_path` without setting `trial_dir_name`')
        return Path(self.experiment_fs_path, self.trial_dir_name).as_posix()

    @property
    def checkpoint_fs_path(self) -> str:
        """The current checkpoint directory path on the `storage_filesystem`.

        "Current" refers to the checkpoint that is currently being created/persisted.
        The user of this class is responsible for setting the `current_checkpoint_index`
        (e.g., incrementing when needed).
        """
        return Path(self.trial_fs_path, self.checkpoint_dir_name).as_posix()

    @property
    def checkpoint_dir_name(self) -> str:
        """The current checkpoint directory name, based on the checkpoint index."""
        return StorageContext._make_checkpoint_dir_name(self.current_checkpoint_index)

    @staticmethod
    def get_experiment_dir_name(run_obj: Union[str, Callable, Type]) -> str:
        from ray.tune.experiment import Experiment
        from ray.tune.utils import date_str
        run_identifier = Experiment.get_trainable_name(run_obj)
        if bool(int(os.environ.get('TUNE_DISABLE_DATED_SUBDIR', 0))):
            dir_name = run_identifier
        else:
            dir_name = '{}_{}'.format(run_identifier, date_str())
        return dir_name

    @staticmethod
    def _make_checkpoint_dir_name(index: int):
        """Get the name of the checkpoint directory, given an index."""
        return f'checkpoint_{index:06d}'