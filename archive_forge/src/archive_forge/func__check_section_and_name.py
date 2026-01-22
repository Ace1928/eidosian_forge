import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _check_section_and_name(self, section: SectionLike, name: NameLike) -> Tuple[Section, Name]:
    if not isinstance(section, tuple):
        section = (section,)
    checked_section = tuple([subsection.encode(self.encoding) if not isinstance(subsection, bytes) else subsection for subsection in section])
    if not isinstance(name, bytes):
        name = name.encode(self.encoding)
    return (checked_section, name)