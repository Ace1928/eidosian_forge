from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def _warn_missing_dry_run_prefix(line, warn_missing_dry_run_prefix, warn_function):
    if warn_missing_dry_run_prefix and warn_function:
        warn_function('Event line is missing dry-run mode marker: {0!r}. Please report this at https://github.com/ansible-collections/community.docker/issues/new?assignees=&labels=&projects=&template=bug_report.md'.format(line))