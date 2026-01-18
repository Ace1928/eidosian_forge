from unittest import mock
from oslo_serialization import jsonutils
import requests
import webob
from keystonemiddleware import ec2_token
from keystonemiddleware.tests.unit import utils
This represents a WSGI app protected by the auth_token middleware.