import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def write_listings(package_name, filename, output_format):
    """Write listing information into a file.

    Parameters
    ----------
    package_name : str
        Name of the package to inspect.
    filename : str
        Output filename. Always overwrite.
    output_format : str
        Support formats are "html" and "rst".
    """
    package = __import__(package_name)
    if hasattr(package, '__path__'):
        mods = list_modules_in_package(package)
    else:
        mods = [package]
    if output_format == 'html':
        with open(filename + '.html', 'w') as fout:
            fmtr = HTMLFormatter(fileobj=fout)
            _format_module_infos(fmtr, package_name, mods)
    elif output_format == 'rst':
        with open(filename + '.rst', 'w') as fout:
            fmtr = ReSTFormatter(fileobj=fout)
            _format_module_infos(fmtr, package_name, mods)
    else:
        raise ValueError("Output format '{}' is not supported".format(output_format))