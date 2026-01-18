import logging
import os
import re
from typing import List, Optional, Tuple
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import CommandArgs, make_command
from pip._internal.vcs.versioncontrol import (
Return options to be used on calls to Subversion that contact the server.

        These options are applicable for the following ``svn`` subcommands used
        in this class.

            - checkout
            - switch
            - update

        :return: A list of command line arguments to pass to ``svn``.
        