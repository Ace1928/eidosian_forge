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
def _AddActions(actions_to_add, spec, relative_path_of_gyp_file):
    actions = spec.get('actions', [])
    have_setup_env = set()
    for a in actions:
        inputs = a.get('inputs') or [relative_path_of_gyp_file]
        attached_to = inputs[0]
        need_setup_env = attached_to not in have_setup_env
        cmd = _BuildCommandLineForRule(spec, a, has_input_path=False, do_setup_env=need_setup_env)
        have_setup_env.add(attached_to)
        _AddActionStep(actions_to_add, inputs=inputs, outputs=a.get('outputs', []), description=a.get('message', a['action_name']), command=cmd)