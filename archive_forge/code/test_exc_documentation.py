import testtools
from unittest import mock
from glanceclient import exc
exc.from_response should not print HTML.