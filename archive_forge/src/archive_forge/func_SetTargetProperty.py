import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def SetTargetProperty(output, target_name, property_name, values, sep=''):
    """Given a target, sets the given property."""
    output.write('set_target_properties(')
    output.write(target_name)
    output.write(' PROPERTIES ')
    output.write(property_name)
    output.write(' "')
    for value in values:
        output.write(CMakeStringEscape(value))
        output.write(sep)
    output.write('")\n')