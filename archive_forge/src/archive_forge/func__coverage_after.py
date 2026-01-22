from distutils import cmd
import distutils.errors
import logging
import os
import sys
import warnings
def _coverage_after(self):
    logger.debug('_coverage_after called')
    os.system('coverage combine')
    os.system('coverage html -d ./cover %s' % self.omit)
    os.system('coverage xml -o ./cover/coverage.xml %s' % self.omit)