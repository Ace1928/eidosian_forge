from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
Spec for date revisions:
          date:value
          value can be 'yesterday', 'today', 'tomorrow' or a YYYY-MM-DD string.
          matches the first entry after a given date (either at midnight or
          at a specified time).
        