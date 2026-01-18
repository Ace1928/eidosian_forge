import io
import os
import pathlib
import re
import sys
from pprint import pformat
from IPython.core import magic_arguments
from IPython.core import oinspect
from IPython.core import page
from IPython.core.alias import AliasError, Alias
from IPython.core.error import UsageError
from IPython.core.magic import  (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.openpy import source_to_unicode
from IPython.utils.process import abbrev_cwd
from IPython.utils.terminal import set_term_title
from traitlets import Bool
from warnings import warn
@line_magic
def unalias(self, parameter_s=''):
    """Remove an alias"""
    aname = parameter_s.strip()
    try:
        self.shell.alias_manager.undefine_alias(aname)
    except ValueError as e:
        print(e)
        return
    stored = self.shell.db.get('stored_aliases', {})
    if aname in stored:
        print('Removing %stored alias', aname)
        del stored[aname]
        self.shell.db['stored_aliases'] = stored