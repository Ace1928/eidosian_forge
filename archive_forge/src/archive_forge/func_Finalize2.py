import filecmp
import gyp.common
import gyp.xcodeproj_file
import gyp.xcode_ninja
import errno
import os
import sys
import posixpath
import re
import shutil
import subprocess
import tempfile
def Finalize2(self, xcode_targets, xcode_target_to_target_dict):
    for bf_tgt in self.build_file_dict['targets']:
        if int(bf_tgt.get('xcode_create_dependents_test_runner', 0)):
            tgt_name = bf_tgt['target_name']
            toolset = bf_tgt['toolset']
            qualified_target = gyp.common.QualifiedTarget(self.gyp_path, tgt_name, toolset)
            xcode_target = xcode_targets[qualified_target]
            if isinstance(xcode_target, gyp.xcodeproj_file.PBXAggregateTarget):
                all_run_tests = []
                pbxtds = xcode_target.GetProperty('dependencies')
                for pbxtd in pbxtds:
                    pbxcip = pbxtd.GetProperty('targetProxy')
                    dependency_xct = pbxcip.GetProperty('remoteGlobalIDString')
                    if hasattr(dependency_xct, 'test_runner'):
                        all_run_tests.append(dependency_xct.test_runner)
                if len(all_run_tests) > 0:
                    run_all_target = gyp.xcodeproj_file.PBXAggregateTarget({'name': 'Run %s Tests' % tgt_name, 'productName': tgt_name}, parent=self.project)
                    for run_test_target in all_run_tests:
                        run_all_target.AddDependency(run_test_target)
                    idx = self.project._properties['targets'].index(xcode_target)
                    self.project._properties['targets'].insert(idx + 1, run_all_target)
    for other_pbxproject in self.project._other_pbxprojects.keys():
        self.project.AddOrGetProjectReference(other_pbxproject)
    self.project.SortRemoteProductReferences()
    self.project_file.ComputeIDs()
    self.project_file.EnsureNoIDCollisions()