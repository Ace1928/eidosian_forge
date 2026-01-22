import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteDependencyOnExtraOutputs(self, target, extra_outputs):
    self.WriteMakeRule([self.output_binary], extra_outputs, comment='Build our special outputs first.', order_only=True)