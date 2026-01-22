import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def CreateCMakeTargetName(self, qualified_target):
    base_name = CreateCMakeTargetBaseName(qualified_target)
    if base_name in self.cmake_target_base_names_conficting:
        return CreateCMakeTargetFullName(qualified_target)
    return base_name