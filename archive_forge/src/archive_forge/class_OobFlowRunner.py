from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
class OobFlowRunner(FlowRunner):
    """A flow runner to run OobFlow."""

    def _CreateFlow(self):
        from googlecloudsdk.core.credentials import flow as c_flow
        return c_flow.OobFlow.from_client_config(self._client_config, self._scopes, autogenerate_code_verifier=not properties.VALUES.auth.disable_code_verifier.GetBool())