import logging; log = logging.getLogger(__name__)
import warnings
from passlib import hash
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
passlib.tests.test_handlers - tests for passlib hash algorithms