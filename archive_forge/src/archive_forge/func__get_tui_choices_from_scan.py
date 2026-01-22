import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
def _get_tui_choices_from_scan(repos: Iterable[CachedRepoInfo], preselected: List[str]) -> List:
    """Build a list of choices from the scanned repos.

    Args:
        repos (*Iterable[`CachedRepoInfo`]*):
            List of scanned repos on which we want to delete revisions.
        preselected (*List[`str`]*):
            List of revision hashes that will be preselected.

    Return:
        The list of choices to pass to `inquirer.checkbox`.
    """
    choices: List[Union[Choice, Separator]] = []
    choices.append(Choice(_CANCEL_DELETION_STR, name='None of the following (if selected, nothing will be deleted).', enabled=False))
    for repo in sorted(repos, key=_repo_sorting_order):
        choices.append(Separator(f'\n{repo.repo_type.capitalize()} {repo.repo_id} ({repo.size_on_disk_str}, used {repo.last_accessed_str})'))
        for revision in sorted(repo.revisions, key=_revision_sorting_order):
            choices.append(Choice(revision.commit_hash, name=f'{revision.commit_hash[:8]}: {', '.join(sorted(revision.refs)) or '(detached)'} # modified {revision.last_modified_str}', enabled=revision.commit_hash in preselected))
    return choices