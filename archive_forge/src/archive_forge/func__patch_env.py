from os import environ
import shlex
import subprocess
import sys
import pytest
def _patch_env(*filtered_keys, **kw):
    env = {k: v for k, v in environ.items() if k not in filtered_keys}
    env.update(kw)
    return env