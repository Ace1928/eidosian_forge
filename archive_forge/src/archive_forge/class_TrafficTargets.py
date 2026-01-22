from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
class TrafficTargets(collections_abc.MutableMapping):
    """Wraps a repeated TrafficTarget message and provides dict-like access.

  The dictionary key is one of
     LATEST_REVISION_KEY for the latest revision
     TrafficTarget.revisionName for TrafficTargets with a revision name.

  The dictionary value is a list of all traffic targets referencing the same
  revision, either by name or the latest revision.
  """

    def __init__(self, messages_module, to_wrap):
        """Constructs a new TrafficTargets instance.

    The TrafficTargets instance wraps the to_wrap argument, which is a repeated
    proto message. Operations that mutate to_wrap will usually occur through
    this class, but that is not a requirement. Callers can directly mutate
    to_wrap by accessing the proto directly.

    Args:
      messages_module: The message module that defines TrafficTarget.
      to_wrap: The traffic targets to wrap.
    """
        self._messages = messages_module
        self._m = to_wrap
        self._traffic_target_cls = self._messages.TrafficTarget

    def __getitem__(self, key):
        """Gets a sorted list of traffic targets associated with the given key.

    Allows accessing traffic targets based on the revision they reference
    (either directly by name or the latest ready revision by specifying
    "LATEST" as the key).

    Returns a sorted list of traffic targets to support comparison operations on
    TrafficTargets objects which should be independent of the order of the
    traffic targets for a given key.

    Args:
      key: A revision name or "LATEST" to get the traffic targets for.

    Returns:
      A sorted list of traffic targets associated with the given key.

    Raises:
      KeyError: If this object does not contain the given key.
    """
        result = sorted((t for t in self._m if GetKey(t) == key), key=_GetItemSortKey)
        if not result:
            raise KeyError(key)
        return result

    def _OtherTargets(self, key):
        """Gets all targets that do not match the given key."""
        return [t for t in self._m if GetKey(t) != key]

    def __setitem__(self, key, new_targets):
        """Implements evaluation of `self[key] = targets`."""
        if key not in self:
            self._m.extend(new_targets)
        else:
            self._m[:] = self._OtherTargets(key) + new_targets

    def SetPercent(self, key, percent):
        """Set the given percent in the traffic targets.

    Moves any tags on existing targets with the specified key to zero percent
    targets.

    Args:
      key: Name of the revision (or "LATEST") to set the percent for.
      percent: Percent of traffic to set.
    """
        existing = self.get(key)
        if existing:
            new_targets = [NewTrafficTarget(self._messages, key, tag=t.tag) for t in existing if t.tag]
            new_targets.append(NewTrafficTarget(self._messages, key, percent))
            self[key] = new_targets
        else:
            self._m.append(NewTrafficTarget(self._messages, key, percent))

    def __delitem__(self, key):
        """Implements evaluation of `del self[key]`."""
        if key not in self:
            raise KeyError(key)
        self._m[:] = self._OtherTargets(key)

    def __contains__(self, key):
        """Implements evaluation of `item in self`."""
        for target in self._m:
            if key == GetKey(target):
                return True
        return False

    @property
    def _key_set(self):
        """A set containing the mapping's keys."""
        return set((GetKey(t) for t in self._m))

    def __len__(self):
        """Implements evaluation of `len(self)`."""
        return len(self._key_set)

    def __iter__(self):
        """Returns an iterator over the traffic target keys."""
        return iter(self._key_set)

    def MakeSerializable(self):
        return self._m

    def __repr__(self):
        content = ', '.join(('{}: {}'.format(k, v) for k, v in self.items()))
        return '[%s]' % content

    def _GetNormalizedTraffic(self):
        """Returns normalized targets, split into percent and tags targets.

    Moves all tags to 0% targets. Combines all targets with a non-zero percent
    that reference the same revision into a single target. Drops 0% targets
    without tags. Does not modify the underlying repeated message field.

    Returns:
      A tuple of (percent targets, tag targets), where percent targets is a
      dictionary mapping key to traffic target for all targets with percent
      greater than zero, and tag targets is a list of traffic targets with
      tags and percent equal to zero.
    """
        tag_targets = []
        percent_targets = {}
        for target in self._m:
            key = GetKey(target)
            if target.tag:
                tag_targets.append(NewTrafficTarget(self._messages, key, tag=target.tag))
            if target.percent:
                percent_targets.setdefault(key, NewTrafficTarget(self._messages, key, 0)).percent += target.percent
        return (percent_targets, tag_targets)

    def _ValidateCurrentTraffic(self, existing_percent_targets):
        """Validate current traffic targets."""
        percent = 0
        for target in existing_percent_targets:
            percent += target.percent
        if percent != 100:
            raise ValueError('Current traffic allocation of %s is not 100 percent' % percent)
        for target in existing_percent_targets:
            if target.percent < 0:
                raise ValueError('Current traffic for target %s is negative (%s)' % (GetKey(target), target.percent))

    def _GetUnassignedTargets(self, new_percentages):
        """Get TrafficTargets with traffic not in new_percentages."""
        result = {}
        for target in self._m:
            key = GetKey(target)
            if target.percent and key not in new_percentages:
                result[key] = target
        return result

    def _ValidateNewPercentages(self, new_percentages, unspecified_targets):
        """Validate the new traffic percentages the user specified."""
        specified_percent = sum(new_percentages.values())
        if specified_percent > 100:
            raise InvalidTrafficSpecificationError('Over 100% of traffic is specified.')
        for key in new_percentages:
            if new_percentages[key] < 0 or new_percentages[key] > 100:
                raise InvalidTrafficSpecificationError('New traffic for target %s is %s, not between 0 and 100' % (key, new_percentages[key]))
        if not unspecified_targets and specified_percent < 100:
            raise InvalidTrafficSpecificationError('Every target with traffic is updated but 100% of traffic has not been specified.')

    def _GetPercentUnspecifiedTraffic(self, new_percentages):
        """Returns percentage of traffic not explicitly specified by caller."""
        specified_percent = sum(new_percentages.values())
        return 100 - specified_percent

    def _IntPercentages(self, float_percentages):
        """Returns rounded integer percentages."""
        rounded_percentages = {k: int(float_percentages[k]) for k in float_percentages}
        loss = int(round(sum(float_percentages.values()))) - sum(rounded_percentages.values())
        correction_precedence = sorted(float_percentages.items(), key=NewRoundingCorrectionPrecedence)
        for key, _ in correction_precedence[:loss]:
            rounded_percentages[key] += 1
        return rounded_percentages

    def _GetAssignedPercentages(self, new_percentages, unassigned_targets):
        percent_to_assign = self._GetPercentUnspecifiedTraffic(new_percentages)
        if percent_to_assign == 0:
            return {}
        percent_to_assign_from = sum((target.percent for target in unassigned_targets.values()))
        assigned_percentages = {}
        for k in unassigned_targets:
            assigned_percentages[k] = unassigned_targets[k].percent * float(percent_to_assign) / percent_to_assign_from
        return assigned_percentages

    def UpdateTraffic(self, new_percentages: Mapping[str, int]):
        """Update traffic percent assignments.

    The updated traffic percent assignments will include assignments explicitly
    specified by the caller. If the caller does not assign 100% of
    traffic explicitly this function will scale traffic for targets
    the user does not specify with an existing percent greater than zero up or
    down based on the provided assignments as needed.

    This method normalizes the traffic targets while updating the traffic
    percent assignments. Normalization merges all targets referencing the same
    revision without tags into a single target with the combined percent.
    Normalization also moves any tags referencing a revision to zero percent
    targets.

    The update removes targets with 0% traffic unless:
     o The user explicitly specifies under 100% of total traffic
     o The user does not explicitly specify 0% traffic for the target.
     o The 0% target has a tag.

    Args:
      new_percentages: Map from revision to percent traffic for the revision.
        'LATEST' means the latest rev.

    Raises:
      ValueError: If the current traffic for the service is invalid.
      InvalidTrafficSpecificationError: If the caller attempts to set
        the traffic for the service to an incorrect state.
    """
        existing_percent_targets, tag_targets = self._GetNormalizedTraffic()
        self._ValidateCurrentTraffic(existing_percent_targets.values())
        updated_percentages = new_percentages.copy()
        unassigned_targets = self._GetUnassignedTargets(updated_percentages)
        self._ValidateNewPercentages(updated_percentages, unassigned_targets)
        updated_percentages.update(self._GetAssignedPercentages(updated_percentages, unassigned_targets))
        int_percentages = self._IntPercentages(updated_percentages)
        new_percent_targets = []
        for key in int_percentages:
            if key in new_percentages and new_percentages[key] == 0:
                continue
            elif key in existing_percent_targets:
                target = existing_percent_targets[key]
                target.percent = int_percentages[key]
            else:
                target = NewTrafficTarget(self._messages, key, int_percentages[key])
            new_percent_targets.append(target)
        new_percent_targets = sorted(new_percent_targets, key=SortKeyFromTarget)
        del self._m[:]
        self._m.extend(new_percent_targets)
        self._m.extend(tag_targets)

    def ZeroLatestTraffic(self, latest_ready_revision_name):
        """Reasign traffic from LATEST to the current latest revision."""
        percent_targets, tag_targets = self._GetNormalizedTraffic()
        if LATEST_REVISION_KEY in percent_targets:
            latest = percent_targets.pop(LATEST_REVISION_KEY)
            if latest_ready_revision_name in percent_targets:
                percent_targets[latest_ready_revision_name].percent += latest.percent
            else:
                percent_targets[latest_ready_revision_name] = NewTrafficTarget(self._messages, latest_ready_revision_name, latest.percent)
            sorted_percent_targets = sorted(percent_targets.values(), key=SortKeyFromTarget)
            self._m[:] = sorted_percent_targets + tag_targets

    def TagToKey(self):
        return {target.tag: GetKey(target) for target in self._m if target.tag}

    def UpdateTags(self, to_update: Mapping[str, str], to_remove: Container[str], clear_others: bool):
        """Update traffic tags.

    Removes and/or clears existing traffic tags as requested. Always adds new
    tags to zero percent targets for the specified revision. Treats a tag
    update as a remove and add.

    Args:
      to_update: A dictionary mapping tag to revision name or 'LATEST' for the
        latest ready revision.
      to_remove: A list of tags to remove.
      clear_others: A boolean indicating whether to clear tags not specified in
        to_update.
    """
        new_targets = []
        if not self._m:
            self._m[:] = [NewTrafficTarget(self._messages, LATEST_REVISION_KEY, 100)]
        for target in self._m:
            if clear_others or target.tag in to_remove or target.tag in to_update:
                target.tag = None
            if target.percent or target.tag:
                new_targets.append(target)
        for tag, revision_key in sorted(to_update.items()):
            new_targets.append(NewTrafficTarget(self._messages, revision_key, tag=tag))
        self._m[:] = new_targets