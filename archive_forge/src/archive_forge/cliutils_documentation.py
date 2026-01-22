import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
Check and add fields to output columns.

    If there is any value in fields that not an attribute of obj,
    CommandError will be raised.
    If fields has duplicate values (case sensitive), we will make them unique
    and ignore duplicate ones.
    :param fields: A list of string contains the fields to be printed.
    :param objs: An list of object which will be used to check if field is
                 valid or not. Note, we don't check fields if obj is None or
                 empty.
    :param exclude_fields: A tuple of string which contains the fields to be
                           excluded.
    :param filters: A dictionary defines how to get value from fields, this
                    is useful when field's value is a complex object such as
                    dictionary.
    :return: columns, formatters.
             columns is a list of string which will be used as table header.
             formatters is a dictionary specifies how to display the value
             of the field.
             They can be [], {}.
    :raise: magnumclient.common.apiclient.exceptions.CommandError.
    