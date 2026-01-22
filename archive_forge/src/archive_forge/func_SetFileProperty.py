import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def SetFileProperty(output, source_name, property_name, values, sep):
    """Given a set of source file, sets the given property on them."""
    output.write('set_source_files_properties(')
    output.write(source_name)
    output.write(' PROPERTIES ')
    output.write(property_name)
    output.write(' "')
    for value in values:
        output.write(CMakeStringEscape(value))
        output.write(sep)
    output.write('")\n')