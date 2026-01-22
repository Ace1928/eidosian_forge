import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
Initialize EngineFacade using oslo.config config instance options.

        :param conf: oslo.config config instance
        :type conf: oslo_config.cfg.ConfigOpts

        :param sqlite_fk: enable foreign keys in SQLite
        :type sqlite_fk: bool

        :param expire_on_commit: expire session objects on commit
        :type expire_on_commit: bool

        