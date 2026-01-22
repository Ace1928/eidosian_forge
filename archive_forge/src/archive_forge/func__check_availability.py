import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
def _check_availability(self, provisioner_name: str) -> bool:
    """
        Checks that the given provisioner is available.

        If the given provisioner is not in the current set of loaded provisioners an attempt
        is made to fetch the named entry point and, if successful, loads it into the cache.

        :param provisioner_name:
        :return:
        """
    is_available = True
    if provisioner_name not in self.provisioners:
        try:
            ep = self._get_provisioner(provisioner_name)
            self.provisioners[provisioner_name] = ep
        except Exception:
            is_available = False
    return is_available