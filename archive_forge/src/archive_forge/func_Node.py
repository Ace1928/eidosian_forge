from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def Node(command=None, commands=None, constraints=None, flags=None, path=None, positionals=None, description=None):
    """Creates and returns a CLI tree node dict."""
    path = []
    if command:
        path.append(command)
        if not description:
            description = 'The {} command.'.format(command)
    return {LOOKUP_CAPSULE: '', LOOKUP_COMMANDS: commands or {}, LOOKUP_CONSTRAINTS: constraints or {}, LOOKUP_FLAGS: flags or {}, LOOKUP_IS_GROUP: True, LOOKUP_IS_HIDDEN: False, LOOKUP_PATH: path, LOOKUP_POSITIONALS: positionals or {}, LOOKUP_RELEASE: 'GA', LOOKUP_SECTIONS: {'DESCRIPTION': description}}