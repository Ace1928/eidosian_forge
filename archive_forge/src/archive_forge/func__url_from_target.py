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
def _url_from_target(self, target):
    if isinstance(target, Connectable):
        return target.engine.url
    elif isinstance(target, str):
        if '://' not in target:
            target_url = sa_url.make_url('%s://' % target)
        else:
            target_url = sa_url.make_url(target)
        return target_url
    elif isinstance(target, sa_url.URL):
        return target
    else:
        raise ValueError('Invalid target type: %r' % target)