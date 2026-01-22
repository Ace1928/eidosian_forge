from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
class ComponentSnapshotDiff(object):
    """Provides the ability to compare two ComponentSnapshots.

  This class is used to see how the current state-of-the-word compares to what
  we have installed.  It can be for informational purposes (to list available
  updates) but also to determine specifically what components need to be
  uninstalled / installed for a specific update command.

  Attributes:
    current: ComponentSnapshot, The current snapshot state.
    latest: CompnentSnapshot, The new snapshot that is being compared.
  """
    DARWIN_X86_64 = platforms.Platform(platforms.OperatingSystem.MACOSX, platforms.Architecture.x86_64)
    ROSETTA2_FILES = ['/Library/Apple/System/Library/LaunchDaemons/com.apple.oahd.plist', '/Library/Apple/usr/share/rosetta/rosetta', '/Library/Apple/System/Library/Receipts/com.apple.pkg.RosettaUpdateAuto.bom', '/Library/Apple/System/Library/Receipts/com.apple.pkg.RosettaUpdateAuto.plist']

    def __init__(self, current, latest, platform_filter=None):
        """Creates a new diff between two ComponentSnapshots.

    Args:
      current: The current ComponentSnapshot
      latest: The ComponentSnapshot representing a new state we can move to
      platform_filter: platforms.Platform, A platform that components must
        match in order to be considered for any operations.
    """
        self.current = current
        self.latest = latest
        self.__platform_filter = platform_filter
        self.__enable_fallback = self._EnableFallback()
        self.__all_components = current.AllComponentIdsMatching(platform_filter) | latest.AllComponentIdsMatching(platform_filter)
        if self.__enable_fallback:
            self.__all_darwin_x86_64_components = current.AllComponentIdsMatching(self.DARWIN_X86_64) | latest.AllComponentIdsMatching(self.DARWIN_X86_64)
            self.__darwin_x86_64_components = self.__all_darwin_x86_64_components - self.__all_components
            self.__native_all_components = set(self.__all_components)
            self.__all_components |= self.__darwin_x86_64_components
            self.__diffs = [ComponentDiff(component_id, current, latest, platform_filter=platform_filter) for component_id in self.__native_all_components]
            self.__diffs.extend([ComponentDiff(component_id, current, latest, platform_filter=self.DARWIN_X86_64) for component_id in self.__darwin_x86_64_components])
        else:
            self.__diffs = [ComponentDiff(component_id, current, latest, platform_filter=platform_filter) for component_id in self.__all_components]
        self.__removed_components = set((diff.id for diff in self.__diffs if diff.state is ComponentState.REMOVED))
        self.__new_components = set((diff.id for diff in self.__diffs if diff.state is ComponentState.NEW))
        self.__updated_components = set((diff.id for diff in self.__diffs if diff.state is ComponentState.UPDATE_AVAILABLE))

    def _EnableFallback(self):
        return self.__platform_filter and self.__platform_filter.operating_system == platforms.OperatingSystem.MACOSX and (self.__platform_filter.architecture == platforms.Architecture.arm)

    def InvalidUpdateSeeds(self, component_ids):
        """Sees if any of the given components don't exist locally or remotely.

    Args:
      component_ids: list of str, The components that the user wants to update.

    Returns:
      set of str, The component ids that do not exist anywhere.
    """
        invalid_seeds = set(component_ids) - self.__all_components
        missing_platform = self.latest.CheckMissingPlatformExecutable(component_ids, self.__platform_filter)
        if self._EnableFallback():
            missing_platform_x86_64 = self.latest.CheckMissingPlatformExecutable(component_ids, self.DARWIN_X86_64)
            missing_platform &= missing_platform_x86_64
            native_invalid_ids = set(component_ids) - self.__native_all_components
            arm_x86_ids = native_invalid_ids & self.__darwin_x86_64_components
            if arm_x86_ids:
                rosetta2_installed = self._CheckRosetta2Exists()
                if not rosetta2_installed:
                    log.warning('The ARM versions of the components [{}] are not available yet. To download and execute the x86_64 version of the components, please install Rosetta 2 first by running the command: softwareupdate --install-rosetta.'.format(', '.join(arm_x86_ids)))
                    invalid_seeds |= arm_x86_ids
        if missing_platform:
            log.warning('The platform specific binary does not exist for components [{}].'.format(', '.join(missing_platform)))
        return invalid_seeds | missing_platform

    def _CheckRosetta2Exists(self):
        for path in self.ROSETTA2_FILES:
            if os.path.isfile(path):
                return True
        return False

    def AllDiffs(self):
        """Gets all ComponentDiffs for this snapshot comparison.

    Returns:
      The list of all ComponentDiffs between the snapshots.
    """
        return self._FilterDiffs(None)

    def AvailableUpdates(self):
        """Gets ComponentDiffs for components where there is an update available.

    Returns:
      The list of ComponentDiffs.
    """
        return self._FilterDiffs(ComponentState.UPDATE_AVAILABLE)

    def AvailableToInstall(self):
        """Gets ComponentDiffs for new components that can be installed.

    Returns:
      The list of ComponentDiffs.
    """
        return self._FilterDiffs(ComponentState.NEW)

    def Removed(self):
        """Gets ComponentDiffs for components that no longer exist.

    Returns:
      The list of ComponentDiffs.
    """
        return self._FilterDiffs(ComponentState.REMOVED)

    def UpToDate(self):
        """Gets ComponentDiffs for installed components that are up to date.

    Returns:
      The list of ComponentDiffs.
    """
        return self._FilterDiffs(ComponentState.UP_TO_DATE)

    def _FilterDiffs(self, state):
        if not state:
            filtered = self.__diffs
        else:
            filtered = [diff for diff in self.__diffs if diff.state is state]
        return sorted(filtered, key=lambda d: d.name)

    def FilterDuplicatesArm(self, component_ids):
        """Filter out x86_64 components that are available in arm versions."""
        return set((i for i in component_ids if not ('darwin-x86_64' in i and i.replace('x86_64', 'arm') in component_ids)))

    def ToRemove(self, update_seed):
        """Calculate the components that need to be uninstalled.

    Based on this given set of components, determine what we need to remove.
    When an update is done, we update all components connected to the initial
    set.  Based on this, we need to remove things that have been updated, or
    that no longer exist.  This method works with ToInstall().  For a given
    update set the update process should remove anything from ToRemove()
    followed by installing everything in ToInstall().  It is possible (and
    likely) that a component will be in both of these sets (when a new version
    is available).

    Args:
      update_seed: list of str, The component ids that we want to update.

    Returns:
      set of str, The component ids that should be removed.
    """
        if self._EnableFallback():
            connected = self.current.ConnectedComponents(update_seed, platform_filter=self.__platform_filter)
            connected |= self.latest.ConnectedComponents(connected | set(update_seed), platform_filter=self.__platform_filter)
            connected_darwin_x86_64 = self.current.ConnectedComponents(update_seed, platform_filter=self.DARWIN_X86_64)
            connected_darwin_x86_64 |= self.latest.ConnectedComponents(connected_darwin_x86_64 | set(update_seed), platform_filter=self.DARWIN_X86_64)
            connected |= connected_darwin_x86_64
            x86_removal_candidates = connected - self.FilterDuplicatesArm(connected)
            installed_components = set(self.current.components.keys())
            x86_removal_seed = x86_removal_candidates & installed_components
            if x86_removal_seed:
                log.warning('The ARM versions of the following components are available, replacing installed x86_64 versions: [{}].'.format(', '.join(x86_removal_seed)))
            removal_candidates = connected & set(self.current.components.keys())
            return (self.__removed_components | self.__updated_components | x86_removal_seed) & removal_candidates
        else:
            connected = self.current.ConnectedComponents(update_seed, platform_filter=self.__platform_filter)
            connected |= self.latest.ConnectedComponents(connected | set(update_seed), platform_filter=self.__platform_filter)
            removal_candidates = connected & set(self.current.components.keys())
            return (self.__removed_components | self.__updated_components) & removal_candidates

    def ToInstall(self, update_seed):
        """Calculate the components that need to be installed.

    Based on this given set of components, determine what we need to install.
    When an update is done, we update all components connected to the initial
    set.  Based on this, we need to install things that have been updated or
    that are new.  This method works with ToRemove().  For a given update set
    the update process should remove anything from ToRemove() followed by
    installing everything in ToInstall().  It is possible (and likely) that a
    component will be in both of these sets (when a new version is available).

    Args:
      update_seed: list of str, The component ids that we want to update.

    Returns:
      set of str, The component ids that should be removed.
    """
        installed_components = list(self.current.components.keys())
        missing_platform = self.latest.CheckMissingPlatformExecutable(update_seed, self.__platform_filter)
        if self._EnableFallback():
            missing_platform_darwin_x86_64 = self.latest.CheckMissingPlatformExecutable(update_seed, self.DARWIN_X86_64)
            native_valid_seed = self.__native_all_components - missing_platform
            native_seed = set(update_seed) & native_valid_seed
            darwin_x86_64 = set(update_seed) - native_seed
            darwin_x86_64 -= missing_platform_darwin_x86_64
            valid_seed = native_seed | darwin_x86_64
            platform_seeds = [c_id for c_id in darwin_x86_64 if 'darwin' not in c_id]
            if platform_seeds:
                log.warning('The ARM versions of the following components are not available yet, using x86_64 versions instead: [{}].'.format(', '.join(platform_seeds)))
            local_connected = self.current.ConnectedComponents(valid_seed, platform_filter=self.__platform_filter)
            all_required = self.latest.DependencyClosureForComponents(local_connected | set(valid_seed), platform_filter=self.__platform_filter)
            local_connected_darwin_x86_64 = self.current.ConnectedComponents(valid_seed, platform_filter=self.DARWIN_X86_64)
            all_required |= self.latest.DependencyClosureForComponents(local_connected_darwin_x86_64 | valid_seed, platform_filter=self.DARWIN_X86_64)
            remote_connected = self.latest.ConnectedComponents(local_connected | valid_seed, platform_filter=self.__platform_filter)
            remote_connected |= self.latest.ConnectedComponents(local_connected_darwin_x86_64 | valid_seed, platform_filter=self.__platform_filter)
            all_required |= remote_connected & set(installed_components)
            all_required = self.FilterDuplicatesArm(all_required)
            dep_missing_platform = self.latest.CheckMissingPlatformExecutable(all_required, self.DARWIN_X86_64)
            if dep_missing_platform:
                log.warning('The platform specific binary does not exist for components [{}].'.format(', '.join(dep_missing_platform)))
                all_required -= dep_missing_platform
        else:
            local_connected = self.current.ConnectedComponents(update_seed, platform_filter=self.__platform_filter)
            all_required = self.latest.DependencyClosureForComponents(local_connected | set(update_seed), platform_filter=self.__platform_filter)
            remote_connected = self.latest.ConnectedComponents(local_connected | set(update_seed), platform_filter=self.__platform_filter)
            all_required |= remote_connected & set(installed_components)
            dep_missing_platform = self.latest.CheckMissingPlatformExecutable(all_required, self.__platform_filter)
            if dep_missing_platform:
                log.warning('The platform specific binary does not exist for components [{}].'.format(', '.join(dep_missing_platform)))
                all_required -= dep_missing_platform
        different = self.__new_components | self.__updated_components
        return set((c for c in all_required if c in different or c not in installed_components))

    def DetailsForCurrent(self, component_ids):
        """Gets the schema.Component objects for all ids from the current snapshot.

    Args:
      component_ids: list of str, The component ids to get.

    Returns:
      A list of schema.Component objects sorted by component display name.
    """
        return sorted(self.current.ComponentsFromIds(component_ids), key=lambda c: c.details.display_name)

    def DetailsForLatest(self, component_ids):
        """Gets the schema.Component objects for all ids from the latest snapshot.

    Args:
      component_ids: list of str, The component ids to get.

    Returns:
      A list of schema.Component objects sorted by component display name.
    """
        return sorted(self.latest.ComponentsFromIds(component_ids), key=lambda c: c.details.display_name)