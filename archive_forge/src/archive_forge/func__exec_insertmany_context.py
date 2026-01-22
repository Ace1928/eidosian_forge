from __future__ import annotations
import itertools
import random
import re
import sys
import sqlalchemy as sa
from .base import TestBase
from .. import config
from .. import mock
from ..assertions import eq_
from ..assertions import ne_
from ..util import adict
from ..util import drop_all_tables_from_metadata
from ... import event
from ... import util
from ...schema import sort_tables_and_constraints
from ...sql import visitors
from ...sql.elements import ClauseElement
def _exec_insertmany_context(dialect, context):
    with mock.patch.object(dialect, '_deliver_insertmanyvalues_batches', new=_deliver_insertmanyvalues_batches):
        return orig_conn(dialect, context)