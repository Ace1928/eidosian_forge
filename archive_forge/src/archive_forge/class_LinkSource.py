import logging
import mimetypes
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from pip._vendor.packaging.utils import (
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url, url_to_path
from pip._internal.vcs import is_url
class LinkSource:

    @property
    def link(self) -> Optional[Link]:
        """Returns the underlying link, if there's one."""
        raise NotImplementedError()

    def page_candidates(self) -> FoundCandidates:
        """Candidates found by parsing an archive listing HTML file."""
        raise NotImplementedError()

    def file_links(self) -> FoundLinks:
        """Links found by specifying archives directly."""
        raise NotImplementedError()