import gyp.common
import gyp.xcode_emulation
import json
import os
def IsMac(params):
    return 'mac' == gyp.common.GetFlavor(params)