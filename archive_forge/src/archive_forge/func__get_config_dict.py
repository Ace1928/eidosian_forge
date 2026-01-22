import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@classmethod
def _get_config_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    token = kwargs.pop('token', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    trust_remote_code = kwargs.pop('trust_remote_code', None)
    subfolder = kwargs.pop('subfolder', '')
    from_pipeline = kwargs.pop('_from_pipeline', None)
    from_auto_class = kwargs.pop('_from_auto', False)
    commit_hash = kwargs.pop('_commit_hash', None)
    if trust_remote_code is True:
        logger.warning('The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.')
    user_agent = {'file_type': 'config', 'from_auto_class': from_auto_class}
    if from_pipeline is not None:
        user_agent['using_pipeline'] = from_pipeline
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
        resolved_config_file = pretrained_model_name_or_path
        is_local = True
    elif is_remote_url(pretrained_model_name_or_path):
        configuration_file = pretrained_model_name_or_path
        resolved_config_file = download_url(pretrained_model_name_or_path)
    else:
        configuration_file = kwargs.pop('_configuration_file', CONFIG_NAME)
        try:
            resolved_config_file = cached_file(pretrained_model_name_or_path, configuration_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, subfolder=subfolder, _commit_hash=commit_hash)
            commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
        except EnvironmentError:
            raise
        except Exception:
            raise EnvironmentError(f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a {configuration_file} file")
    try:
        config_dict = cls._dict_from_json_file(resolved_config_file)
        config_dict['_commit_hash'] = commit_hash
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise EnvironmentError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.")
    if is_local:
        logger.info(f'loading configuration file {resolved_config_file}')
    else:
        logger.info(f'loading configuration file {configuration_file} from cache at {resolved_config_file}')
    if 'auto_map' in config_dict and (not is_local):
        config_dict['auto_map'] = add_model_info_to_auto_map(config_dict['auto_map'], pretrained_model_name_or_path)
    return (config_dict, kwargs)