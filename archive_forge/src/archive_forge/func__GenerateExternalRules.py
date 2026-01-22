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
def _GenerateExternalRules(rules, output_dir, spec, sources, options, actions_to_add):
    """Generate an external makefile to do a set of rules.

  Arguments:
    rules: the list of rules to include
    output_dir: path containing project and gyp files
    spec: project specification data
    sources: set of sources known
    options: global generator options
    actions_to_add: The list of actions we will add to.
  """
    filename = '{}_rules{}.mk'.format(spec['target_name'], options.suffix)
    mk_file = gyp.common.WriteOnDiff(os.path.join(output_dir, filename))
    mk_file.write('OutDirCygwin:=$(shell cygpath -u "$(OutDir)")\n')
    mk_file.write('IntDirCygwin:=$(shell cygpath -u "$(IntDir)")\n')
    all_inputs = OrderedSet()
    all_outputs = OrderedSet()
    all_output_dirs = OrderedSet()
    first_outputs = []
    for rule in rules:
        trigger_files = _FindRuleTriggerFiles(rule, sources)
        for tf in trigger_files:
            inputs, outputs = _RuleInputsAndOutputs(rule, tf)
            all_inputs.update(OrderedSet(inputs))
            all_outputs.update(OrderedSet(outputs))
            first_outputs.append(list(outputs)[0])
            output_dirs = [os.path.split(i)[0] for i in outputs]
            for od in output_dirs:
                all_output_dirs.add(od)
    first_outputs_cyg = [_Cygwinify(i) for i in first_outputs]
    mk_file.write('all: %s\n' % ' '.join(first_outputs_cyg))
    for od in all_output_dirs:
        if od:
            mk_file.write('\tmkdir -p `cygpath -u "%s"`\n' % od)
    mk_file.write('\n')
    for rule in rules:
        trigger_files = _FindRuleTriggerFiles(rule, sources)
        for tf in trigger_files:
            inputs, outputs = _RuleInputsAndOutputs(rule, tf)
            inputs = [_Cygwinify(i) for i in inputs]
            outputs = [_Cygwinify(i) for i in outputs]
            cmd = [_RuleExpandPath(c, tf) for c in rule['action']]
            cmd = ['"%s"' % i for i in cmd]
            cmd = ' '.join(cmd)
            mk_file.write('{}: {}\n'.format(' '.join(outputs), ' '.join(inputs)))
            mk_file.write('\t%s\n\n' % cmd)
    mk_file.close()
    sources.add(filename)
    cmd = ['make', 'OutDir=$(OutDir)', 'IntDir=$(IntDir)', '-j', '${NUMBER_OF_PROCESSORS_PLUS_1}', '-f', filename]
    cmd = _BuildCommandLineForRuleRaw(spec, cmd, True, False, True, True)
    all_inputs = list(all_inputs)
    all_inputs.insert(0, filename)
    _AddActionStep(actions_to_add, inputs=_FixPaths(all_inputs), outputs=_FixPaths(all_outputs), description='Running external rules for %s' % spec['target_name'], command=cmd)