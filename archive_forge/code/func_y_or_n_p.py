from __future__ import print_function
import os
import platform
import remove_pyreadline
import setuptools.command.easy_install as easy_install
import setuptools.package_index
import shutil
import sys
def y_or_n_p(prompt):
    response = raw_input('%s (y/n) ' % (prompt,)).strip().lower()
    while response not in ['y', 'n']:
        response = raw_input('  Please answer y or n: ').strip().lower()
    return response