import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
def _unexpected_status(path, response):
    raise errors.UnexpectedHttpStatus(path, response.status, response.data.decode('utf-8', 'replace'), headers=response.getheaders())