from typing import List, Optional
from sys import version_info
from importlib import reload, metadata
from collections import defaultdict
import dataclasses
import re
from semantic_version import Version
def _refresh_compilers():
    """Scan installed PennyLane compiler packages to refresh the compilers
    names and entry points.
    """
    AvailableCompilers.names_entrypoints = defaultdict(dict)
    entries = defaultdict(dict, metadata.entry_points())['pennylane.compilers'] if version_info[:2] == (3, 9) else metadata.entry_points(group='pennylane.compilers')
    for entry in entries:
        try:
            compiler_name, e_name = entry.name.split('.')
            AvailableCompilers.names_entrypoints[compiler_name][e_name] = entry
        except ValueError:
            compiler_name = entry.module.split('.')[0]
            AvailableCompilers.names_entrypoints[compiler_name][entry.name] = entry
    for _, eps_dict in AvailableCompilers.names_entrypoints.items():
        ep_interface = AvailableCompilers.entrypoints_interface
        if any((ep not in eps_dict.keys() for ep in ep_interface)):
            raise KeyError(f'expected {ep_interface}, but recieved {eps_dict}')