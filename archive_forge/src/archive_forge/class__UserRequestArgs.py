from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
class _UserRequestArgs:
    """Class contains user flags and should be passed to RequestConfig factory.

  Should not be mutated while being passed around. See RequestConfig classes
  for "Attributes" docstring. Specifics depend on API client.
  """

    def __init__(self, gzip_settings=None, manifest_path=None, no_clobber=None, override_unlocked_retention=None, precondition_generation_match=None, precondition_metageneration_match=None, predefined_acl_string=None, predefined_default_object_acl_string=None, preserve_posix=None, preserve_symlinks=None, resource_args=None):
        """Sets properties."""
        self.gzip_settings = gzip_settings
        self.manifest_path = os.path.expanduser(manifest_path) if manifest_path else None
        self.no_clobber = no_clobber
        self.override_unlocked_retention = override_unlocked_retention
        self.precondition_generation_match = precondition_generation_match
        self.precondition_metageneration_match = precondition_metageneration_match
        self.predefined_acl_string = predefined_acl_string
        self.predefined_default_object_acl_string = predefined_default_object_acl_string
        self.preserve_posix = preserve_posix
        self.preserve_symlinks = preserve_symlinks
        self.resource_args = resource_args

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.gzip_settings == other.gzip_settings and self.manifest_path == other.manifest_path and (self.no_clobber == other.no_clobber) and (self.override_unlocked_retention == other.override_unlocked_retention) and (self.precondition_generation_match == other.precondition_generation_match) and (self.precondition_metageneration_match == other.precondition_metageneration_match) and (self.predefined_acl_string == other.predefined_acl_string) and (self.predefined_default_object_acl_string == other.predefined_default_object_acl_string) and (self.preserve_posix == other.preserve_posix) and (self.preserve_symlinks == other.preserve_symlinks) and (self.resource_args == other.resource_args)

    def __repr__(self):
        return debug_output.generic_repr(self)