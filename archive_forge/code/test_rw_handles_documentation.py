import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
Tests for ImageReadHandle.