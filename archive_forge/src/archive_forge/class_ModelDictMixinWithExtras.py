import datetime
import functools
import pytz
from oslo_db import exception as db_exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import models
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from osprofiler import opts as profiler
import osprofiler.sqlalchemy
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.orm.attributes import flag_modified, InstrumentedAttribute
from sqlalchemy import types as sql_types
from keystone.common import driver_hints
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class ModelDictMixinWithExtras(models.ModelBase):
    """Mixin making model behave with dict-like interfaces includes extras.

    NOTE: DO NOT USE THIS FOR FUTURE SQL MODELS. "Extra" column is a legacy
          concept that should not be carried forward with new SQL models
          as the concept of "arbitrary" properties is not in line with
          the design philosophy of Keystone.
    """
    attributes = []
    _msg = 'Programming Error: Model does not have an "extra" column. Unless the model already has an "extra" column and has existed in a previous released version of keystone with the extra column included, the model should use "ModelDictMixin" instead.'

    @classmethod
    def from_dict(cls, d):
        new_d = d.copy()
        if not hasattr(cls, 'extra'):
            raise AttributeError(cls._msg)
        new_d['extra'] = {k: new_d.pop(k) for k in d.keys() if k not in cls.attributes and k != 'extra'}
        return cls(**new_d)

    def to_dict(self, include_extra_dict=False):
        """Return the model's attributes as a dictionary.

        If include_extra_dict is True, 'extra' attributes are literally
        included in the resulting dictionary twice, for backwards-compatibility
        with a broken implementation.

        """
        if not hasattr(self, 'extra'):
            raise AttributeError(self._msg)
        d = self.extra.copy()
        for attr in self.__class__.attributes:
            d[attr] = getattr(self, attr)
        if include_extra_dict:
            d['extra'] = self.extra.copy()
        return d

    def __getitem__(self, key):
        """Evaluate if key is in extra or not, to return correct item."""
        if key in self.extra:
            return self.extra[key]
        return getattr(self, key)