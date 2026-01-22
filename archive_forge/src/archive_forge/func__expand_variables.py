import fnmatch
import getpass
import os
import re
import shlex
import socket
from hashlib import sha1
from io import StringIO
from functools import partial
from .ssh_exception import CouldNotCanonicalize, ConfigParseError
def _expand_variables(self, config, target_hostname):
    """
        Return a dict of config options with expanded substitutions
        for a given original & current target hostname.

        Please refer to :doc:`/api/config` for details.

        :param dict config: the currently parsed config
        :param str hostname: the hostname whose config is being looked up
        """
    for k in config:
        if config[k] is None:
            continue
        tokenizer = partial(self._tokenize, config, target_hostname, k)
        if isinstance(config[k], list):
            for i, value in enumerate(config[k]):
                config[k][i] = tokenizer(value)
        else:
            config[k] = tokenizer(config[k])
    return config