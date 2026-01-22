import errno
import os
import re
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import commands, errors, option, osutils, registry, trace
class ResolveActionOption(option.RegistryOption):

    def __init__(self):
        super().__init__('action', 'How to resolve the conflict.', value_switches=True, registry=resolve_action_registry)