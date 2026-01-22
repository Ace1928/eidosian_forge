from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re

        Replace 'site_type' key with 'type' in the config.

        Parameters:
            config (list or dict) - Configuration details.

        Returns:
            updated_config (list or dict) - Updated config after replacing the keys.
        