import os
import shutil
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.hashutil import md5_file_hex
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.wb_value import WBValue
class _SavedModel(WBValue, Generic[SavedModelObjType]):
    """Internal W&B Artifact model storage.

    _model_type_id: (str) The id of the SavedModel subclass used to serialize the model.
    """
    _log_type: ClassVar[str]
    _path_extension: ClassVar[str]
    _model_obj: Optional['SavedModelObjType']
    _path: Optional[str]
    _input_obj_or_path: Union[SavedModelObjType, str]

    def __init__(self, obj_or_path: Union[SavedModelObjType, str], **kwargs: Any) -> None:
        super().__init__()
        if self.__class__ == _SavedModel:
            raise TypeError('Cannot instantiate abstract SavedModel class - please use SavedModel.init(...) instead.')
        self._model_obj = None
        self._path = None
        self._input_obj_or_path = obj_or_path
        input_is_path = isinstance(obj_or_path, str) and os.path.exists(obj_or_path)
        if input_is_path:
            assert isinstance(obj_or_path, str)
            self._set_obj(self._deserialize(obj_or_path))
        else:
            self._set_obj(obj_or_path)
        self._copy_to_disk()
        if not input_is_path:
            self._unset_obj()

    @staticmethod
    def init(obj_or_path: Any, **kwargs: Any) -> '_SavedModel':
        maybe_instance = _SavedModel._maybe_init(obj_or_path, **kwargs)
        if maybe_instance is None:
            raise ValueError(f'No suitable SavedModel subclass constructor found for obj_or_path: {obj_or_path}')
        return maybe_instance

    @classmethod
    def from_json(cls: Type['_SavedModel'], json_obj: dict, source_artifact: 'Artifact') -> '_SavedModel':
        path = json_obj['path']
        entry = source_artifact.manifest.entries.get(path)
        if entry is not None:
            dl_path = str(source_artifact.get_entry(path).download())
        else:
            dl_path = _load_dir_from_artifact(source_artifact, path)
        return cls(dl_path)

    def to_json(self, run_or_artifact: Union['LocalRun', 'Artifact']) -> dict:
        if isinstance(run_or_artifact, wandb.wandb_sdk.wandb_run.Run):
            raise ValueError('SavedModel cannot be added to run - must use artifact')
        artifact = run_or_artifact
        json_obj = {'type': self._log_type}
        assert self._path is not None, 'Cannot add SavedModel to Artifact without path'
        if os.path.isfile(self._path):
            already_added_path = artifact.get_added_local_path_name(self._path)
            if already_added_path is not None:
                json_obj['path'] = already_added_path
            else:
                target_path = os.path.join('.wb_data', 'saved_models', os.path.basename(self._path))
                json_obj['path'] = artifact.add_file(self._path, target_path, True).path
        elif os.path.isdir(self._path):
            json_obj['path'] = _add_deterministic_dir_to_artifact(artifact, self._path, os.path.join('.wb_data', 'saved_models'))
        else:
            raise ValueError(f'Expected a path to a file or directory, got {self._path}')
        return json_obj

    def model_obj(self) -> SavedModelObjType:
        """Return the model object."""
        if self._model_obj is None:
            assert self._path is not None, 'Cannot load model object without path'
            self._set_obj(self._deserialize(self._path))
        assert self._model_obj is not None, 'Model object is None'
        return self._model_obj

    @staticmethod
    def _deserialize(path: str) -> SavedModelObjType:
        """Return the model object from a path. Allowed to throw errors."""
        raise NotImplementedError

    @staticmethod
    def _validate_obj(obj: Any) -> bool:
        """Validate the model object. Allowed to throw errors."""
        raise NotImplementedError

    @staticmethod
    def _serialize(obj: SavedModelObjType, dir_or_file_path: str) -> None:
        """Save the model to disk.

        The method will receive a directory path which all files needed for
        deserialization should be saved. A directory will always be passed if
        _path_extension is an empty string, else a single file will be passed. Allowed
        to throw errors.
        """
        raise NotImplementedError

    @classmethod
    def _maybe_init(cls: Type['_SavedModel'], obj_or_path: Any, **kwargs: Any) -> Optional['_SavedModel']:
        try:
            return cls(obj_or_path, **kwargs)
        except Exception as e:
            if DEBUG_MODE:
                print(f'{cls}._maybe_init({obj_or_path}) failed: {e}')
            pass
        for child_cls in cls.__subclasses__():
            maybe_instance = child_cls._maybe_init(obj_or_path, **kwargs)
            if maybe_instance is not None:
                return maybe_instance
        return None

    @classmethod
    def _tmp_path(cls: Type['_SavedModel']) -> str:
        assert isinstance(cls._path_extension, str), '_path_extension must be a string'
        tmp_path = os.path.abspath(os.path.join(MEDIA_TMP.name, runid.generate_id()))
        if cls._path_extension != '':
            tmp_path += '.' + cls._path_extension
        return tmp_path

    def _copy_to_disk(self) -> None:
        tmp_path = self._tmp_path()
        self._dump(tmp_path)
        self._path = tmp_path

    def _unset_obj(self) -> None:
        assert self._path is not None, 'Cannot unset object if path is None'
        self._model_obj = None

    def _set_obj(self, model_obj: Any) -> None:
        assert model_obj is not None and self._validate_obj(model_obj), f'Invalid model object {model_obj}'
        self._model_obj = model_obj

    def _dump(self, target_path: str) -> None:
        assert self._model_obj is not None, 'Cannot dump if model object is None'
        self._serialize(self._model_obj, target_path)