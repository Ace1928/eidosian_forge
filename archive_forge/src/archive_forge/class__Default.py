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
class _Default:
    """Mark a value as a default value.

    A value in the local configuration dictionary wrapped with
    _Default() will not take precedence over a value that is specified
    in cfg.CONF.   Values that are set after the fact using configure()
    will supersede those in cfg.CONF.
    """
    __slots__ = ('value',)
    _notset = _symbol('NOTSET')

    def __init__(self, value=_notset):
        self.value = value

    @classmethod
    def resolve(cls, value):
        if isinstance(value, _Default):
            v = value.value
            if v is cls._notset:
                return None
            else:
                return v
        else:
            return value

    @classmethod
    def resolve_w_conf(cls, value, conf, key):
        if isinstance(value, _Default):
            v = getattr(conf.database, key, value.value)
            if v is cls._notset:
                return None
            else:
                return v
        else:
            return value

    @classmethod
    def is_set(cls, value):
        if not isinstance(value, _Default):
            return True
        return value.value is not cls._notset

    @classmethod
    def is_set_w_conf(cls, value, conf, key):
        if hasattr(conf.database, key):
            opt = conf.database._group._opts[key]['opt']
            group = conf.database._group
            if opt.deprecated_for_removal and conf.get_location(key, group=group.name).location == cfg.Locations.opt_default:
                return False
            return True
        return cls.is_set(value)