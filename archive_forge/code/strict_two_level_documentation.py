from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
from keystone.limit.models import base
Check the input limits satisfy the related project tree or not.

        1. Ensure the input is legal.
        2. Ensure the input will not break the exist limit tree.

        