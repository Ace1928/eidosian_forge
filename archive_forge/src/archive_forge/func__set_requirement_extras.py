import copy
import logging
import os
import re
from typing import Collection, Dict, List, Optional, Set, Tuple, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.specifiers import Specifier
from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import is_url, vcs
def _set_requirement_extras(req: Requirement, new_extras: Set[str]) -> Requirement:
    """
    Returns a new requirement based on the given one, with the supplied extras. If the
    given requirement already has extras those are replaced (or dropped if no new extras
    are given).
    """
    match: Optional[re.Match[str]] = re.fullmatch('([\\w\\t .-]+)(\\[[^\\]]*\\])?(.*)', str(req), flags=re.ASCII)
    assert match is not None, f'regex match on requirement {req} failed, this should never happen'
    pre: Optional[str] = match.group(1)
    post: Optional[str] = match.group(3)
    assert pre is not None and post is not None, f'regex group selection for requirement {req} failed, this should never happen'
    extras: str = '[%s]' % ','.join(sorted(new_extras)) if new_extras else ''
    return Requirement(f'{pre}{extras}{post}')