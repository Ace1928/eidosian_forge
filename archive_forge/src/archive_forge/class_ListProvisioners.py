from __future__ import annotations
import errno
import json
import os.path
import sys
import typing as t
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Dict, Instance, List, Unicode
from traitlets.config.application import Application
from . import __version__
from .kernelspec import KernelSpecManager
from .provisioning.factory import KernelProvisionerFactory
class ListProvisioners(JupyterApp):
    """An app to list provisioners."""
    version = __version__
    description = 'List available provisioners for use in kernel specifications.'

    def start(self) -> None:
        """Start the application."""
        kfp = KernelProvisionerFactory.instance(parent=self)
        print('Available kernel provisioners:')
        provisioners = kfp.get_provisioner_entries()
        name_len = len(sorted(provisioners, key=lambda name: len(name))[-1])
        for name in sorted(provisioners):
            print(f'  {name.ljust(name_len)}    {provisioners[name]}')