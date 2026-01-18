import os
import tempfile
from xml import sax
from six import StringIO
from boto import handler
from boto.exception import GSResponseError
from boto.gs.acl import ACL
from tests.integration.gs.testcase import GSTestCase
Integration tests for GS versioning support.