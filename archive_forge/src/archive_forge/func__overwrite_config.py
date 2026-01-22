import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def _overwrite_config(raw_config):
    config_path = _get_path()
    if not os.path.exists(config_path):
        file_descriptor = os.open(config_path, os.O_CREAT | os.O_RDWR, 384)
        os.close(file_descriptor)
    if not os.stat(config_path).st_mode == 33152:
        os.chmod(config_path, 384)
    with open(config_path, 'w') as cfg:
        raw_config.write(cfg)