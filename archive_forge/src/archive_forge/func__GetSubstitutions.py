import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def _GetSubstitutions(self, bundle_identifier, app_identifier_prefix):
    """Constructs a dictionary of variable substitutions for Entitlements.plist.

    Args:
      bundle_identifier: string, value of CFBundleIdentifier from Info.plist
      app_identifier_prefix: string, value for AppIdentifierPrefix

    Returns:
      Dictionary of substitutions to apply when generating Entitlements.plist.
    """
    return {'CFBundleIdentifier': bundle_identifier, 'AppIdentifierPrefix': app_identifier_prefix}