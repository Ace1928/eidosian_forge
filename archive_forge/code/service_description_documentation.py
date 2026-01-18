import typing as ty
import warnings
import os_service_types
from openstack import _log
from openstack import exceptions
from openstack import proxy as proxy_mod
from openstack import warnings as os_warnings
Create a Proxy for the service in question.

        :param instance:
          The `openstack.connection.Connection` we're working with.
        