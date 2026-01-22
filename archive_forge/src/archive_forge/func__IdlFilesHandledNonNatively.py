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
def _IdlFilesHandledNonNatively(spec, sources):
    using_idl = False
    for rule in spec.get('rules', []):
        if rule['extension'] == 'idl' and int(rule.get('msvs_external_rule', 0)):
            using_idl = True
            break
    if using_idl:
        excluded_idl = [i for i in sources if i.endswith('.idl')]
    else:
        excluded_idl = []
    return excluded_idl