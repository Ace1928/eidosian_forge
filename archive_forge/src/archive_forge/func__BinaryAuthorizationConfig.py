from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator, List
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
def _BinaryAuthorizationConfig(self, existing_binauthz=None) -> fleet_messages.BinaryAuthorizationConfig:
    """Construct binauthz config from args."""
    new_binauthz = self.messages.BinaryAuthorizationConfig()
    new_binauthz.evaluationMode = self._EvaluationMode()
    new_binauthz.policyBindings = list(self._PolicyBindings())
    if existing_binauthz is None:
        ret = new_binauthz
    else:
        ret = existing_binauthz
        if new_binauthz.evaluationMode is not None:
            ret.evaluationMode = new_binauthz.evaluationMode
        if new_binauthz.policyBindings is not None:
            ret.policyBindings = new_binauthz.policyBindings
    if ret.policyBindings and (not ret.evaluationMode):
        raise exceptions.InvalidArgumentException('--binauthz-policy-bindings', _PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='binauthz-evaluation-mode', opt='binauthz-policy-bindings'))
    if ret.evaluationMode == fleet_messages.BinaryAuthorizationConfig.EvaluationModeValueValuesEnum.DISABLED:
        ret.policyBindings = []
    return self.TrimEmpty(ret)