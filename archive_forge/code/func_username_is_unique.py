import getpass
import os
import sys
from django.contrib.auth import get_user_model
from django.contrib.auth.management import get_default_username
from django.contrib.auth.password_validation import validate_password
from django.core import exceptions
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS
from django.utils.functional import cached_property
from django.utils.text import capfirst
@cached_property
def username_is_unique(self):
    if self.username_field.unique:
        return True
    return any((len(unique_constraint.fields) == 1 and unique_constraint.fields[0] == self.username_field.name for unique_constraint in self.UserModel._meta.total_unique_constraints))