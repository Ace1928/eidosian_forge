from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible.module_utils.common.text.converters import to_native
def update_from_swarm_info(self, swarm_info):
    spec = swarm_info['Spec']
    ca_config = spec.get('CAConfig') or dict()
    if self.node_cert_expiry is None:
        self.node_cert_expiry = ca_config.get('NodeCertExpiry')
    if self.ca_force_rotate is None:
        self.ca_force_rotate = ca_config.get('ForceRotate')
    dispatcher = spec.get('Dispatcher') or dict()
    if self.dispatcher_heartbeat_period is None:
        self.dispatcher_heartbeat_period = dispatcher.get('HeartbeatPeriod')
    raft = spec.get('Raft') or dict()
    if self.snapshot_interval is None:
        self.snapshot_interval = raft.get('SnapshotInterval')
    if self.keep_old_snapshots is None:
        self.keep_old_snapshots = raft.get('KeepOldSnapshots')
    if self.heartbeat_tick is None:
        self.heartbeat_tick = raft.get('HeartbeatTick')
    if self.log_entries_for_slow_followers is None:
        self.log_entries_for_slow_followers = raft.get('LogEntriesForSlowFollowers')
    if self.election_tick is None:
        self.election_tick = raft.get('ElectionTick')
    orchestration = spec.get('Orchestration') or dict()
    if self.task_history_retention_limit is None:
        self.task_history_retention_limit = orchestration.get('TaskHistoryRetentionLimit')
    encryption_config = spec.get('EncryptionConfig') or dict()
    if self.autolock_managers is None:
        self.autolock_managers = encryption_config.get('AutoLockManagers')
    if self.name is None:
        self.name = spec['Name']
    if self.labels is None:
        self.labels = spec.get('Labels') or {}
    if 'LogDriver' in spec['TaskDefaults']:
        self.log_driver = spec['TaskDefaults']['LogDriver']