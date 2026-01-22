import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
@classmethod
def all_impls(cls):
    """Return an iterator of all possible BackendImpl objects.

        These are BackendImpls that are implemented, but not
        necessarily provisionable.

        """
    for database_type in cls.impl.reg:
        if database_type == '*':
            continue
        yield BackendImpl.impl(database_type)