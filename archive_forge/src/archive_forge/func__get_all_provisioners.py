import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
@staticmethod
def _get_all_provisioners() -> List[EntryPoint]:
    """Wrapper around entry_points (to fetch the set of provisioners) - primarily to facilitate testing."""
    return entry_points(group=KernelProvisionerFactory.GROUP_NAME)