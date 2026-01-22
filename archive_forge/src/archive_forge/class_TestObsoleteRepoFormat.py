import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
class TestObsoleteRepoFormat(RepositoryFormat2a):

    @classmethod
    def get_format_string(cls):
        return b'Test Obsolete Repository Format'

    def is_deprecated(self):
        return True