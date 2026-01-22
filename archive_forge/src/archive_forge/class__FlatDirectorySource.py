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
class _FlatDirectorySource(LinkSource):
    """Link source specified by ``--find-links=<path-to-dir>``.

    This looks the content of the directory, and returns:

    * ``page_candidates``: Links listed on each HTML file in the directory.
    * ``file_candidates``: Archives in the directory.
    """
    _paths_to_urls: Dict[str, _FlatDirectoryToUrls] = {}

    def __init__(self, candidates_from_page: CandidatesFromPage, path: str, project_name: str) -> None:
        self._candidates_from_page = candidates_from_page
        self._project_name = canonicalize_name(project_name)
        if path in self._paths_to_urls:
            self._path_to_urls = self._paths_to_urls[path]
        else:
            self._path_to_urls = _FlatDirectoryToUrls(path=path)
            self._paths_to_urls[path] = self._path_to_urls

    @property
    def link(self) -> Optional[Link]:
        return None

    def page_candidates(self) -> FoundCandidates:
        for url in self._path_to_urls.page_candidates:
            yield from self._candidates_from_page(Link(url))

    def file_links(self) -> FoundLinks:
        for url in self._path_to_urls.project_name_to_urls[self._project_name]:
            yield Link(url)