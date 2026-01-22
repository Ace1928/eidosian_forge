import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteSubMake(self, output_filename, makefile_path, targets, build_dir):
    """Write a "sub-project" Makefile.

        This is a small, wrapper Makefile that calls the top-level Makefile to build
        the targets from a single gyp file (i.e. a sub-project).

        Arguments:
          output_filename: sub-project Makefile name to write
          makefile_path: path to the top-level Makefile
          targets: list of "all" targets for this sub-project
          build_dir: build output directory, relative to the sub-project
        """
    gyp.common.EnsureDirExists(output_filename)
    self.fp = open(output_filename, 'w')
    self.fp.write(header)
    self.WriteLn('export builddir_name ?= %s' % os.path.join(os.path.dirname(output_filename), build_dir))
    self.WriteLn('.PHONY: all')
    self.WriteLn('all:')
    if makefile_path:
        makefile_path = ' -C ' + makefile_path
    self.WriteLn('\t$(MAKE){} {}'.format(makefile_path, ' '.join(targets)))
    self.fp.close()