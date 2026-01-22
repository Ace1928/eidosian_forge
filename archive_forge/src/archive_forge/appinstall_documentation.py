import types
import os
import string
import uuid
from paste.deploy import appconfig
from paste.script import copydir
from paste.script.command import Command, BadCommand, run as run_command
from paste.script.util import secret
from paste.util import import_string
import paste.script.templates
import pkg_resources

        Called to setup an application, given its configuration
        file/directory.

        The default implementation calls
        ``package.websetup.setup_config(command, filename, section,
        vars)`` or ``package.websetup.setup_app(command, config,
        vars)``

        With ``setup_app`` the ``config`` object is a dictionary with
        the extra attributes ``global_conf``, ``local_conf`` and
        ``filename``
        