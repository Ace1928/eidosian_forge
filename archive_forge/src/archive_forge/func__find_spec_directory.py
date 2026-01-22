from __future__ import annotations
import json
import os
import re
import shutil
import typing as t
import warnings
from jupyter_core.paths import SYSTEM_JUPYTER_PATH, jupyter_data_dir, jupyter_path
from traitlets import Bool, CaselessStrEnum, Dict, HasTraits, List, Set, Type, Unicode, observe
from traitlets.config import LoggingConfigurable
from .provisioning import KernelProvisionerFactory as KPF  # noqa
def _find_spec_directory(self, kernel_name: str) -> str | None:
    """Find the resource directory of a named kernel spec"""
    for kernel_dir in [kd for kd in self.kernel_dirs if os.path.isdir(kd)]:
        files = os.listdir(kernel_dir)
        for f in files:
            path = pjoin(kernel_dir, f)
            if f.lower() == kernel_name and _is_kernel_dir(path):
                return path
    if kernel_name == NATIVE_KERNEL_NAME:
        try:
            from ipykernel.kernelspec import RESOURCES
        except ImportError:
            pass
        else:
            return RESOURCES
    return None