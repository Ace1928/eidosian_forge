import logging
from optparse import Values
from typing import Generator, Iterable, Iterator, List, NamedTuple, Optional
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.utils.misc import write_output
def _get_requiring_packages(current_dist: BaseDistribution) -> Iterator[str]:
    return (dist.metadata['Name'] or 'UNKNOWN' for dist in installed.values() if current_dist.canonical_name in {canonicalize_name(d.name) for d in dist.iter_dependencies()})