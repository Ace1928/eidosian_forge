import os
import mock
import pytest
from ..test_util import TestUtilSSL  # noqa: E402, F401
from ..with_dummyserver.test_https import (  # noqa: E402, F401
from ..with_dummyserver.test_socketlevel import (  # noqa: E402, F401

        If a certificate has two subject alternative names, cryptography raises
        an x509.DuplicateExtension exception.
        