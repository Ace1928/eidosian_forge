import enum
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import os
import os.path
import sys
import traceback
def _get_default_invalidation_mode():
    if os.environ.get('SOURCE_DATE_EPOCH'):
        return PycInvalidationMode.CHECKED_HASH
    else:
        return PycInvalidationMode.TIMESTAMP