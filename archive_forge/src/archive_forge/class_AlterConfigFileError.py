from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.runtimes import python
from googlecloudsdk.api_lib.app.runtimes import python_compat
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class AlterConfigFileError(exceptions.Error):
    """Error when attempting to update an existing config file (app.yaml)."""

    def __init__(self, inner_exception):
        super(AlterConfigFileError, self).__init__('Could not alter app.yaml due to an internal error:\n{0}\nPlease update app.yaml manually.'.format(inner_exception))