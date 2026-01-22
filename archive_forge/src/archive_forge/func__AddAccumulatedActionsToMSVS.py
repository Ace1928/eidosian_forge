import ntpath
import os
import posixpath
import re
import subprocess
import sys
from collections import OrderedDict
import gyp.common
import gyp.easy_xml as easy_xml
import gyp.generator.ninja as ninja_generator
import gyp.MSVSNew as MSVSNew
import gyp.MSVSProject as MSVSProject
import gyp.MSVSSettings as MSVSSettings
import gyp.MSVSToolFile as MSVSToolFile
import gyp.MSVSUserFile as MSVSUserFile
import gyp.MSVSUtil as MSVSUtil
import gyp.MSVSVersion as MSVSVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def _AddAccumulatedActionsToMSVS(p, spec, actions_dict):
    """Add actions accumulated into an actions_dict, merging as needed.

  Arguments:
    p: the target project
    spec: the target project dict
    actions_dict: dictionary keyed on input name, which maps to a list of
        dicts describing the actions attached to that input file.
  """
    for primary_input in actions_dict:
        inputs = OrderedSet()
        outputs = OrderedSet()
        descriptions = []
        commands = []
        for action in actions_dict[primary_input]:
            inputs.update(OrderedSet(action['inputs']))
            outputs.update(OrderedSet(action['outputs']))
            descriptions.append(action['description'])
            commands.append(action['command'])
        description = ', and also '.join(descriptions)
        command = '\r\n'.join(commands)
        _AddCustomBuildToolForMSVS(p, spec, primary_input=primary_input, inputs=inputs, outputs=outputs, description=description, cmd=command)