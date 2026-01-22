import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def WriteActionsRulesCopies(self, spec, extra_sources, prebuild, mac_bundle_depends):
    """Write out the Actions, Rules, and Copies steps.  Return a path
        representing the outputs of these steps."""
    outputs = []
    if self.is_mac_bundle:
        mac_bundle_resources = spec.get('mac_bundle_resources', [])[:]
    else:
        mac_bundle_resources = []
    extra_mac_bundle_resources = []
    if 'actions' in spec:
        outputs += self.WriteActions(spec['actions'], extra_sources, prebuild, extra_mac_bundle_resources)
    if 'rules' in spec:
        outputs += self.WriteRules(spec['rules'], extra_sources, prebuild, mac_bundle_resources, extra_mac_bundle_resources)
    if 'copies' in spec:
        outputs += self.WriteCopies(spec['copies'], prebuild, mac_bundle_depends)
    if 'sources' in spec and self.flavor == 'win':
        outputs += self.WriteWinIdlFiles(spec, prebuild)
    if self.xcode_settings and self.xcode_settings.IsIosFramework():
        self.WriteiOSFrameworkHeaders(spec, outputs, prebuild)
    stamp = self.WriteCollapsedDependencies('actions_rules_copies', outputs)
    if self.is_mac_bundle:
        xcassets = self.WriteMacBundleResources(extra_mac_bundle_resources + mac_bundle_resources, mac_bundle_depends)
        partial_info_plist = self.WriteMacXCassets(xcassets, mac_bundle_depends)
        self.WriteMacInfoPlist(partial_info_plist, mac_bundle_depends)
    return stamp