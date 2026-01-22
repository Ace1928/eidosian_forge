from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
import sys
import time
from apitools.base.py import list_pager
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import times
import six
class TensorflowVersionParser(object):
    """Helper to parse tensorflow versions."""

    class ParseError(Exception):
        """Error raised with input is unabled to be parse as a TF version."""

    class Result(object):
        """Helper to capture result of parsing the TF version."""

        def __init__(self, major=0, minor=0, patch=0, is_nightly=False, modifier=''):
            self.major = major
            self.minor = minor
            self.patch = patch
            self.is_nightly = is_nightly
            self.modifier = modifier

        def IsUnknown(self):
            return self.major == 0 and self.minor == 0 and (not self.is_nightly)

        def VersionString(self):
            if self.is_nightly:
                return 'nightly{}'.format(self.modifier)
            if self.major == 0 and self.minor == 0:
                return self.modifier
            return '{}.{}{}'.format(self.major, self.minor, self.modifier)

        def __hash__(self):
            return hash(self.major) + hash(self.minor) + hash(self.patch) + hash(self.is_nightly) + hash(self.modifier)

        def __eq__(self, other):
            return self.major == other.major and self.minor == other.minor and (self.patch == other.patch) and (self.is_nightly == other.is_nightly) and (self.modifier == other.modifier)

        def __lt__(self, other):
            if not self.is_nightly and (not other.is_nightly) and (not self.IsUnknown()) and (not other.IsUnknown()):
                if self.major != other.major:
                    return self.major > other.major
                if self.minor != other.minor:
                    return self.minor > other.minor
                if self.patch != other.patch:
                    return self.patch > other.patch
                if not self.modifier:
                    return True
                if not other.modifier:
                    return False
            if self.is_nightly and other.is_nightly:
                if not self.modifier:
                    return True
                if not other.modifier:
                    return False
            if self.IsUnknown() and other.IsUnknown():
                return self.modifier < other.modifier
            if self.IsUnknown():
                return False
            if other.IsUnknown():
                return True
            if self.is_nightly:
                return False
            return True
    _VERSION_REGEX = re.compile('^(\\d+)\\.(\\d+)(.*)$')
    _NIGHTLY_REGEX = re.compile('^nightly(.*)$')
    _PATCH_NUMBER_REGEX = re.compile('^\\.(\\d+)$')

    @staticmethod
    def ParseVersion(tf_version):
        """Helper to parse the tensorflow version into it's subcomponents."""
        if not tf_version:
            raise TensorflowVersionParser.ParseError('Bad argument: tf_version is empty')
        version_match = TensorflowVersionParser._VERSION_REGEX.match(tf_version)
        nightly_match = TensorflowVersionParser._NIGHTLY_REGEX.match(tf_version)
        if version_match is None and nightly_match is None:
            return TensorflowVersionParser.Result(modifier=tf_version)
        if version_match is not None and nightly_match is not None:
            raise TensorflowVersionParser.ParseError('TF version error: bad version: {}'.format(tf_version))
        if version_match:
            major = int(version_match.group(1))
            minor = int(version_match.group(2))
            result = TensorflowVersionParser.Result(major=major, minor=minor)
            if version_match.group(3):
                patch_match = TensorflowVersionParser._PATCH_NUMBER_REGEX.match(version_match.group(3))
                if patch_match:
                    matched_patch = int(patch_match.group(1))
                    if matched_patch:
                        result.patch = matched_patch
                else:
                    result.modifier = version_match.group(3)
            return result
        if nightly_match:
            result = TensorflowVersionParser.Result(is_nightly=True)
            if nightly_match.group(1):
                result.modifier = nightly_match.group(1)
            return result