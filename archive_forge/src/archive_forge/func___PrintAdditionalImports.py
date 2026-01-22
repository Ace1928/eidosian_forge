import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def __PrintAdditionalImports(self, imports):
    """Print additional imports needed for protorpc."""
    google_imports = [x for x in imports if 'google' in x]
    other_imports = [x for x in imports if 'google' not in x]
    if other_imports:
        for import_ in sorted(other_imports):
            self.__printer(import_)
        self.__printer()
    if google_imports:
        for import_ in sorted(google_imports):
            self.__printer(import_)
        self.__printer()