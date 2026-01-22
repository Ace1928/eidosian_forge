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
def CreateXCConfigurationList(configuration_names):
    xccl = gyp.xcodeproj_file.XCConfigurationList({'buildConfigurations': []})
    if len(configuration_names) == 0:
        configuration_names = ['Default']
    for configuration_name in configuration_names:
        xcbc = gyp.xcodeproj_file.XCBuildConfiguration({'name': configuration_name})
        xccl.AppendProperty('buildConfigurations', xcbc)
    xccl.SetProperty('defaultConfigurationName', configuration_names[0])
    return xccl