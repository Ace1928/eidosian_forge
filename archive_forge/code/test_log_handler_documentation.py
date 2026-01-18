import logging
import fixtures
import oslo_messaging
from oslo_messaging.notify import log_handler
from oslo_messaging.tests import utils as test_utils
from unittest import mock
Tests for log.PublishErrorsHandler