import logging
import os.path
import pathlib
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path, hide_url
from pip._internal.utils.subprocess import make_command
from pip._internal.vcs.versioncontrol import (
@staticmethod
def _git_remote_to_pip_url(url: str) -> str:
    """
        Convert a remote url from what git uses to what pip accepts.

        There are 3 legal forms **url** may take:

            1. A fully qualified url: ssh://git@example.com/foo/bar.git
            2. A local project.git folder: /path/to/bare/repository.git
            3. SCP shorthand for form 1: git@example.com:foo/bar.git

        Form 1 is output as-is. Form 2 must be converted to URI and form 3 must
        be converted to form 1.

        See the corresponding test test_git_remote_url_to_pip() for examples of
        sample inputs/outputs.
        """
    if re.match('\\w+://', url):
        return url
    if os.path.exists(url):
        return pathlib.PurePath(url).as_uri()
    scp_match = SCP_REGEX.match(url)
    if scp_match:
        return scp_match.expand('ssh://\\1\\2/\\3')
    raise RemoteNotValidError(url)