import collections
from collections import abc
import itertools
import logging
import re
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy.engine import Connectable
from sqlalchemy.engine import url as sa_url
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.sql import text
from sqlalchemy import Table
from oslo_db._i18n import _
from oslo_db import exception
from oslo_db.sqlalchemy import models
def _stable_sorting_order(model, sort_keys):
    """Check whether the sorting order is stable.

    :return: True if it is stable, False if it's not, None if it's impossible
    to determine.
    """
    keys = get_unique_keys(model)
    if keys is None:
        return None
    sort_keys_set = set(sort_keys)
    for unique_keys in keys:
        if unique_keys.issubset(sort_keys_set):
            return True
    return False