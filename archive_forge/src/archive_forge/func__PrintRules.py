from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _PrintRules(rule):
    keys = ['services']
    for key in keys:
        if key in rule.keys():
            log.status.Print(' ' + key + ':')
            for value in rule[key]:
                log.status.Print('  - ' + value)