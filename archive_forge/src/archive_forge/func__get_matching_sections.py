import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def _get_matching_sections(self):
    """Get all sections matching ``location``."""
    no_name_section = None
    all_sections = []
    for _, section in self.store.get_sections():
        if section.id is None:
            no_name_section = section
        else:
            all_sections.append(section)
    filtered_sections = _iter_for_location_by_parts([s.id for s in all_sections], self.location)
    iter_all_sections = iter(all_sections)
    matching_sections = []
    if no_name_section is not None:
        matching_sections.append((0, LocationSection(no_name_section, self.location)))
    for section_id, extra_path, length in filtered_sections:
        while True:
            section = next(iter_all_sections)
            if section_id == section.id:
                section = LocationSection(section, extra_path, self.branch_name)
                matching_sections.append((length, section))
                break
    return matching_sections