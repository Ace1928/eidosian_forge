from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class ManyToManyFieldAccessor(FieldAccessor):

    def __init__(self, model, field, name):
        super(ManyToManyFieldAccessor, self).__init__(model, field, name)
        self.model = field.model
        self.rel_model = field.rel_model
        self.through_model = field.through_model
        src_fks = self.through_model._meta.model_refs[self.model]
        dest_fks = self.through_model._meta.model_refs[self.rel_model]
        if not src_fks:
            raise ValueError('Cannot find foreign-key to "%s" on "%s" model.' % (self.model, self.through_model))
        elif not dest_fks:
            raise ValueError('Cannot find foreign-key to "%s" on "%s" model.' % (self.rel_model, self.through_model))
        self.src_fk = src_fks[0]
        self.dest_fk = dest_fks[0]

    def __get__(self, instance, instance_type=None, force_query=False):
        if instance is not None:
            if not force_query and self.src_fk.backref != '+':
                backref = getattr(instance, self.src_fk.backref)
                if isinstance(backref, list):
                    return [getattr(obj, self.dest_fk.name) for obj in backref]
            src_id = getattr(instance, self.src_fk.rel_field.name)
            if src_id is None and self.field._prevent_unsaved:
                raise ValueError('Cannot get many-to-many "%s" for unsaved instance "%s".' % (self.field, instance))
            return ManyToManyQuery(instance, self, self.rel_model).join(self.through_model).join(self.model).where(self.src_fk == src_id)
        return self.field

    def __set__(self, instance, value):
        src_id = getattr(instance, self.src_fk.rel_field.name)
        if src_id is None and self.field._prevent_unsaved:
            raise ValueError('Cannot set many-to-many "%s" for unsaved instance "%s".' % (self.field, instance))
        query = self.__get__(instance, force_query=True)
        query.add(value, clear_existing=True)