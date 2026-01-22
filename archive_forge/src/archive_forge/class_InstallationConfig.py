from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
class InstallationConfig(object):
    """Loads configuration constants from the core config file.

  Attributes:
    version: str, The version of the core component.
    revision: long, A revision number from a component snapshot.  This is a long
      int but formatted as an actual date in seconds (i.e 20151009132504). It is
      *NOT* seconds since the epoch.
    user_agent: str, The base string of the user agent to use when making API
      calls.
    documentation_url: str, The URL where we can redirect people when they need
      more information.
    release_notes_url: str, The URL where we host a nice looking version of our
      release notes.
    snapshot_url: str, The url for the component manager to look at for updates.
    disable_updater: bool, True to disable the component manager for this
      installation.  We do this for distributions through another type of
      package manager like apt-get.
    disable_usage_reporting: bool, True to disable the sending of usage data by
      default.
    snapshot_schema_version: int, The version of the snapshot schema this code
      understands.
    release_channel: str, The release channel for this Cloud SDK distribution.
    config_suffix: str, A string to add to the end of the configuration
      directory name so that different release channels can have separate
      config.
  """
    REVISION_FORMAT_STRING = '%Y%m%d%H%M%S'

    @staticmethod
    def Load():
        """Initializes the object with values from the config file.

    Returns:
      InstallationSpecificData: The loaded data.
    """
        data = json.loads(encoding.Decode(pkg_resources.GetResource(__name__, 'config.json')))
        return InstallationConfig(**data)

    @staticmethod
    def FormatRevision(time_struct):
        """Formats a given time as a revision string for a component snapshot.

    Args:
      time_struct: time.struct_time, The time you want to format.

    Returns:
      int, A revision number from a component snapshot.  This is a int but
      formatted as an actual date in seconds (i.e 20151009132504).  It is *NOT*
      seconds since the epoch.
    """
        return int(time.strftime(InstallationConfig.REVISION_FORMAT_STRING, time_struct))

    @staticmethod
    def ParseRevision(revision):
        """Parse the given revision into a time.struct_time.

    Args:
      revision: long, A revision number from a component snapshot.  This is a
        long int but formatted as an actual date in seconds (i.e
        20151009132504). It is *NOT* seconds since the epoch.

    Returns:
      time.struct_time, The parsed time.
    """
        return time.strptime(six.text_type(revision), InstallationConfig.REVISION_FORMAT_STRING)

    @staticmethod
    def ParseRevisionAsSeconds(revision):
        """Parse the given revision into seconds since the epoch.

    Args:
      revision: long, A revision number from a component snapshot.  This is a
        long int but formatted as an actual date in seconds (i.e
        20151009132504). It is *NOT* seconds since the epoch.

    Returns:
      int, The number of seconds since the epoch that this revision represents.
    """
        return time.mktime(InstallationConfig.ParseRevision(revision))

    def __init__(self, version, revision, user_agent, documentation_url, release_notes_url, snapshot_url, disable_updater, disable_usage_reporting, snapshot_schema_version, release_channel, config_suffix):
        self.version = version
        self.revision = revision
        self.user_agent = str(user_agent)
        self.documentation_url = str(documentation_url)
        self.release_notes_url = str(release_notes_url)
        self.snapshot_url = str(snapshot_url)
        self.disable_updater = disable_updater
        self.disable_usage_reporting = disable_usage_reporting
        self.snapshot_schema_version = snapshot_schema_version
        self.release_channel = str(release_channel)
        self.config_suffix = str(config_suffix)

    def IsAlternateReleaseChannel(self):
        """Determines if this distribution is using an alternate release channel.

    Returns:
      True if this distribution is not one of the 'stable' release channels,
      False otherwise.
    """
        return self.release_channel != 'rapid'