from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateAdaptiveProtectionConfigWithAutoDeployConfig(client, args, existing_adaptive_protection_config):
    """Returns a SecurityPolicyAdaptiveProtectionConfig message with AutoDeployConfig."""
    messages = client.messages
    adaptive_protection_config = CreateAdaptiveProtectionConfig(client, args, existing_adaptive_protection_config)
    if args.IsSpecified('layer7_ddos_defense_auto_deploy_load_threshold') or args.IsSpecified('layer7_ddos_defense_auto_deploy_confidence_threshold') or args.IsSpecified('layer7_ddos_defense_auto_deploy_impacted_baseline_threshold') or args.IsSpecified('layer7_ddos_defense_auto_deploy_expiration_sec'):
        auto_deploy_config = adaptive_protection_config.autoDeployConfig if adaptive_protection_config.autoDeployConfig is not None else messages.SecurityPolicyAdaptiveProtectionConfigAutoDeployConfig()
        if args.IsSpecified('layer7_ddos_defense_auto_deploy_load_threshold'):
            auto_deploy_config.loadThreshold = args.layer7_ddos_defense_auto_deploy_load_threshold
        if args.IsSpecified('layer7_ddos_defense_auto_deploy_confidence_threshold'):
            auto_deploy_config.confidenceThreshold = args.layer7_ddos_defense_auto_deploy_confidence_threshold
        if args.IsSpecified('layer7_ddos_defense_auto_deploy_impacted_baseline_threshold'):
            auto_deploy_config.impactedBaselineThreshold = args.layer7_ddos_defense_auto_deploy_impacted_baseline_threshold
        if args.IsSpecified('layer7_ddos_defense_auto_deploy_expiration_sec'):
            auto_deploy_config.expirationSec = args.layer7_ddos_defense_auto_deploy_expiration_sec
        adaptive_protection_config.autoDeployConfig = auto_deploy_config
    return adaptive_protection_config