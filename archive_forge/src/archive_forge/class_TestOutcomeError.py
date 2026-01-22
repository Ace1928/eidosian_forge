from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
class TestOutcomeError(core_exceptions.Error):
    """The Tool Results backend did not return a valid test outcome."""

    def __init__(self, msg):
        super(TestOutcomeError, self).__init__(msg, exit_code=INFRASTRUCTURE_ERR)