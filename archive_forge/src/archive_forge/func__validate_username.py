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
def _validate_username(self, username, verbose_field_name, database):
    """Validate username. If invalid, return a string error message."""
    if self.username_is_unique:
        try:
            self.UserModel._default_manager.db_manager(database).get_by_natural_key(username)
        except self.UserModel.DoesNotExist:
            pass
        else:
            return 'Error: That %s is already taken.' % verbose_field_name
    if not username:
        return '%s cannot be blank.' % capfirst(verbose_field_name)
    try:
        self.username_field.clean(username, None)
    except exceptions.ValidationError as e:
        return '; '.join(e.messages)