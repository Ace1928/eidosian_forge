from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def _update_template_cfg(self, poco_cfg: messages.Message, state: str) -> messages.Message:
    policy_content = self._extract_policy_content(poco_cfg)
    new_cfg = self.messages.PolicyControllerTemplateLibraryConfig(installation=self._get_template_install_enum(state))
    policy_content.templateLibrary = new_cfg
    poco_cfg.policycontroller.policyControllerHubConfig.policyContent = policy_content
    return poco_cfg