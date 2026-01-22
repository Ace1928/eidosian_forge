import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
class BaseVersionedFile:
    """Pseudo-VersionedFile skeleton for MultiParent"""

    def __init__(self, snapshot_interval=25, max_snapshots=None):
        self._lines = {}
        self._parents = {}
        self._snapshots = set()
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots

    def versions(self):
        return iter(self._parents)

    def has_version(self, version):
        return version in self._parents

    def do_snapshot(self, version_id, parent_ids):
        """Determine whether to perform a snapshot for this version"""
        if self.snapshot_interval is None:
            return False
        if self.max_snapshots is not None and len(self._snapshots) == self.max_snapshots:
            return False
        if len(parent_ids) == 0:
            return True
        for ignored in range(self.snapshot_interval):
            if len(parent_ids) == 0:
                return False
            version_ids = parent_ids
            parent_ids = []
            for version_id in version_ids:
                if version_id not in self._snapshots:
                    parent_ids.extend(self._parents[version_id])
        else:
            return True

    def add_version(self, lines, version_id, parent_ids, force_snapshot=None, single_parent=False):
        """Add a version to the versionedfile

        :param lines: The list of lines to add.  Must be split on '
'.
        :param version_id: The version_id of the version to add
        :param force_snapshot: If true, force this version to be added as a
            snapshot version.  If false, force this version to be added as a
            diff.  If none, determine this automatically.
        :param single_parent: If true, use a single parent, rather than
            multiple parents.
        """
        if force_snapshot is None:
            do_snapshot = self.do_snapshot(version_id, parent_ids)
        else:
            do_snapshot = force_snapshot
        if do_snapshot:
            self._snapshots.add(version_id)
            diff = MultiParent([NewText(lines)])
        else:
            if single_parent:
                parent_lines = self.get_line_list(parent_ids[:1])
            else:
                parent_lines = self.get_line_list(parent_ids)
            diff = MultiParent.from_lines(lines, parent_lines)
            if diff.is_snapshot():
                self._snapshots.add(version_id)
        self.add_diff(diff, version_id, parent_ids)
        self._lines[version_id] = lines

    def get_parents(self, version_id):
        return self._parents[version_id]

    def make_snapshot(self, version_id):
        snapdiff = MultiParent([NewText(self.cache_version(version_id))])
        self.add_diff(snapdiff, version_id, self._parents[version_id])
        self._snapshots.add(version_id)

    def import_versionedfile(self, vf, snapshots, no_cache=True, single_parent=False, verify=False):
        """Import all revisions of a versionedfile

        :param vf: The versionedfile to import
        :param snapshots: If provided, the revisions to make snapshots of.
            Otherwise, this will be auto-determined
        :param no_cache: If true, clear the cache after every add.
        :param single_parent: If true, omit all but one parent text, (but
            retain parent metadata).
        """
        if not (no_cache or not verify):
            raise ValueError()
        revisions = set(vf.versions())
        total = len(revisions)
        with ui.ui_factory.nested_progress_bar() as pb:
            while len(revisions) > 0:
                added = set()
                for revision in revisions:
                    parents = vf.get_parents(revision)
                    if [p for p in parents if p not in self._parents] != []:
                        continue
                    lines = [a + b' ' + l for a, l in vf.annotate(revision)]
                    if snapshots is None:
                        force_snapshot = None
                    else:
                        force_snapshot = revision in snapshots
                    self.add_version(lines, revision, parents, force_snapshot, single_parent)
                    added.add(revision)
                    if no_cache:
                        self.clear_cache()
                        vf.clear_cache()
                        if verify:
                            if not lines == self.get_line_list([revision])[0]:
                                raise AssertionError()
                            self.clear_cache()
                    pb.update(gettext('Importing revisions'), total - len(revisions) + len(added), total)
                revisions = [r for r in revisions if r not in added]

    def select_snapshots(self, vf):
        """Determine which versions to add as snapshots"""
        build_ancestors = {}
        snapshots = set()
        for version_id in topo_iter(vf):
            potential_build_ancestors = set(vf.get_parents(version_id))
            parents = vf.get_parents(version_id)
            if len(parents) == 0:
                snapshots.add(version_id)
                build_ancestors[version_id] = set()
            else:
                for parent in vf.get_parents(version_id):
                    potential_build_ancestors.update(build_ancestors[parent])
                if len(potential_build_ancestors) > self.snapshot_interval:
                    snapshots.add(version_id)
                    build_ancestors[version_id] = set()
                else:
                    build_ancestors[version_id] = potential_build_ancestors
        return snapshots

    def select_by_size(self, num):
        """Select snapshots for minimum output size"""
        num -= len(self._snapshots)
        new_snapshots = self.get_size_ranking()[-num:]
        return [v for n, v in new_snapshots]

    def get_size_ranking(self):
        """Get versions ranked by size"""
        versions = []
        for version_id in self.versions():
            if version_id in self._snapshots:
                continue
            diff_len = self.get_diff(version_id).patch_len()
            snapshot_len = MultiParent([NewText(self.cache_version(version_id))]).patch_len()
            versions.append((snapshot_len - diff_len, version_id))
        versions.sort()
        return versions

    def import_diffs(self, vf):
        """Import the diffs from another pseudo-versionedfile"""
        for version_id in vf.versions():
            self.add_diff(vf.get_diff(version_id), version_id, vf._parents[version_id])

    def get_build_ranking(self):
        """Return revisions sorted by how much they reduce build complexity"""
        could_avoid = {}
        referenced_by = {}
        for version_id in topo_iter(self):
            could_avoid[version_id] = set()
            if version_id not in self._snapshots:
                for parent_id in self._parents[version_id]:
                    could_avoid[version_id].update(could_avoid[parent_id])
                could_avoid[version_id].update(self._parents)
                could_avoid[version_id].discard(version_id)
            for avoid_id in could_avoid[version_id]:
                referenced_by.setdefault(avoid_id, set()).add(version_id)
        available_versions = list(self.versions())
        ranking = []
        while len(available_versions) > 0:
            available_versions.sort(key=lambda x: len(could_avoid[x]) * len(referenced_by.get(x, [])))
            selected = available_versions.pop()
            ranking.append(selected)
            for version_id in referenced_by[selected]:
                could_avoid[version_id].difference_update(could_avoid[selected])
            for version_id in could_avoid[selected]:
                referenced_by[version_id].difference_update(referenced_by[selected])
        return ranking

    def clear_cache(self):
        self._lines.clear()

    def get_line_list(self, version_ids):
        return [self.cache_version(v) for v in version_ids]

    def cache_version(self, version_id):
        try:
            return self._lines[version_id]
        except KeyError:
            pass
        diff = self.get_diff(version_id)
        lines = []
        reconstructor = _Reconstructor(self, self._lines, self._parents)
        reconstructor.reconstruct_version(lines, version_id)
        self._lines[version_id] = lines
        return lines