from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.run import secrets_mapping
class SecretManagerSecret(object):
    """A secret to be fetched from Secret Manager."""

    def __init__(self, name, versions, mapped_secret=None):
        self.name = name
        self.versions = versions
        self.mapped_secret = mapped_secret

    def __eq__(self, other):
        return self.name == other.name and self.versions == other.versions and (self.mapped_secret == other.mapped_secret)

    def __repr__(self):
        return '<Secret: (name="{}", versions={}, mapped_secret="{}")>'.format(self.name, self.versions, self.mapped_secret)

    def __hash__(self):
        return hash((self.name, self.versions, self.mapped_secret))