from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
class TrackLevelResourceReleaseTrackError(ResourceMapError):
    """Exception for when an attempt to access a releast track of a RT occurs."""

    def __init__(self, attempted_rt, accessed_rt):
        super(TrackLevelResourceReleaseTrackError, self).__init__('Attempted accessing of [{}] track of TrackLevelResourceData[{}]'.format(attempted_rt, accessed_rt))