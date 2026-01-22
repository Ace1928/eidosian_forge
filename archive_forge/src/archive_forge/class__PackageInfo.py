import logging
from optparse import Values
from typing import Generator, Iterable, Iterator, List, NamedTuple, Optional
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.utils.misc import write_output
class _PackageInfo(NamedTuple):
    name: str
    version: str
    location: str
    editable_project_location: Optional[str]
    requires: List[str]
    required_by: List[str]
    installer: str
    metadata_version: str
    classifiers: List[str]
    summary: str
    homepage: str
    project_urls: List[str]
    author: str
    author_email: str
    license: str
    entry_points: List[str]
    files: Optional[List[str]]