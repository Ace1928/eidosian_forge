from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import gslib.tests.testcase as testcase
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
from gslib.utils.constants import UTF8
def _run_requester_pays_test(self, command_list, regex=None):
    """Test a command with a user project.

    Run a command with a user project on a Requester Pays bucket. The command is
    expected to pass because the source bucket is Requester Pays. If a regex
    pattern is supplied, also assert that stdout of the command matches it.
    """
    stdout = self.RunGsUtil(self.user_project_flag + command_list, return_stdout=True)
    if regex:
        if isinstance(regex, bytes):
            regex = regex.decode(UTF8)
        self.assertRegexpMatchesWithFlags(stdout, regex, flags=re.IGNORECASE)