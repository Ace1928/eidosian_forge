from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
Assert that headers are correctly set up when finally called.