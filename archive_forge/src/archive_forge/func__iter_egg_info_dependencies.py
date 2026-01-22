import csv
import email.message
import functools
import json
import logging
import pathlib
import re
import zipfile
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import LegacyVersion, Version
from pip._internal.exceptions import NoneMetadataError
from pip._internal.locations import site_packages, user_site
from pip._internal.models.direct_url import (
from pip._internal.utils.compat import stdlib_pkgs  # TODO: Move definition here.
from pip._internal.utils.egg_link import egg_link_path_from_sys_path
from pip._internal.utils.misc import is_local, normalize_path
from pip._internal.utils.urls import url_to_path
from ._json import msg_to_json
def _iter_egg_info_dependencies(self) -> Iterable[str]:
    """Get distribution dependencies from the egg-info directory.

        To ease parsing, this converts a legacy dependency entry into a PEP 508
        requirement string. Like ``_iter_requires_txt_entries()``, there is code
        in ``importlib.metadata`` that does mostly the same, but not do exactly
        what we need.

        Namely, ``importlib.metadata`` does not normalize the extra name before
        putting it into the requirement string, which causes marker comparison
        to fail because the dist-info format do normalize. This is consistent in
        all currently available PEP 517 backends, although not standardized.
        """
    for entry in self._iter_requires_txt_entries():
        extra = canonicalize_name(entry.extra)
        if extra and entry.marker:
            marker = f'({entry.marker}) and extra == "{extra}"'
        elif extra:
            marker = f'extra == "{extra}"'
        elif entry.marker:
            marker = entry.marker
        else:
            marker = ''
        if marker:
            yield f'{entry.requirement} ; {marker}'
        else:
            yield entry.requirement