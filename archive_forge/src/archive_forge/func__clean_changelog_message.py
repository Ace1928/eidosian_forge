from __future__ import unicode_literals
import distutils.errors
from distutils import log
import errno
import io
import os
import re
import subprocess
import time
import pkg_resources
from pbr import options
from pbr import version
def _clean_changelog_message(msg):
    """Cleans any instances of invalid sphinx wording.

    This escapes/removes any instances of invalid characters
    that can be interpreted by sphinx as a warning or error
    when translating the Changelog into an HTML file for
    documentation building within projects.

    * Escapes '_' which is interpreted as a link
    * Escapes '*' which is interpreted as a new line
    * Escapes '`' which is interpreted as a literal
    """
    msg = msg.replace('*', '\\*')
    msg = msg.replace('_', '\\_')
    msg = msg.replace('`', '\\`')
    return msg