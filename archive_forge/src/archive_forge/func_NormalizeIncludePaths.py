import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
def NormalizeIncludePaths(self, include_paths):
    """Normalize include_paths.
        Convert absolute paths to relative to the Android top directory.

        Args:
          include_paths: A list of unprocessed include paths.
        Returns:
          A list of normalized include paths.
        """
    normalized = []
    for path in include_paths:
        if path[0] == '/':
            path = gyp.common.RelativePath(path, self.android_top_dir)
        normalized.append(path)
    return normalized