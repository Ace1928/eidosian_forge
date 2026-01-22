import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
def _overwrite_or_create_databricks_profile(file_name, profile, profile_name='DEFAULT'):
    """Overwrite or create a profile in the databricks config file.

    Args:
        file_name: string, the file name of the databricks config file, usually `~/.databrickscfg`.
        profile: dict, contains the authentiacation profile information.
        profile_name: string, the name of the profile to be overwritten or created.
    """
    profile_name = f'[{profile_name}]'
    lines = []
    if os.path.exists(file_name):
        with open(file_name) as file:
            lines = file.readlines()
    start_index = -1
    end_index = -1
    for i in range(len(lines)):
        if lines[i].strip() == profile_name:
            start_index = i
            break
    if start_index != -1:
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip() == '' or lines[i].startswith('['):
                end_index = i
                break
        end_index = end_index if end_index != -1 else len(lines)
        del lines[start_index:end_index + 1]
    new_profile = []
    new_profile.append(profile_name + '\n')
    new_profile.append(f'host = {profile['host']}\n')
    if 'token' in profile:
        new_profile.append(f'token = {profile['token']}\n')
    else:
        new_profile.append(f'username = {profile['username']}\n')
        new_profile.append(f'password = {profile['password']}\n')
    new_profile.append('\n')
    lines = new_profile + lines
    with open(file_name, 'w') as file:
        file.writelines(lines)