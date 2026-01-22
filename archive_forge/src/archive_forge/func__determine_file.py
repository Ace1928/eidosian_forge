import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output
def _determine_file(self, options: Values, need_value: bool) -> Optional[Kind]:
    file_options = [key for key, value in ((kinds.USER, options.user_file), (kinds.GLOBAL, options.global_file), (kinds.SITE, options.site_file)) if value]
    if not file_options:
        if not need_value:
            return None
        elif any((os.path.exists(site_config_file) for site_config_file in get_configuration_files()[kinds.SITE])):
            return kinds.SITE
        else:
            return kinds.USER
    elif len(file_options) == 1:
        return file_options[0]
    raise PipError('Need exactly one file to operate upon (--user, --site, --global) to perform.')