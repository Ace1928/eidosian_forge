import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
def _add_session(self, session, start_info, groups_by_name):
    """Adds a new Session protobuffer to the 'groups_by_name' dictionary.

        Called by _build_session_groups when we encounter a new session. Creates
        the Session protobuffer and adds it to the relevant group in the
        'groups_by_name' dict. Creates the session group if this is the first time
        we encounter it.

        Args:
          session: api_pb2.Session. The session to add.
          start_info: The SessionStartInfo protobuffer associated with the session.
          groups_by_name: A str to SessionGroup protobuffer dict. Representing the
            session groups and sessions found so far.
        """
    group_name = start_info.group_name or session.name
    if group_name in groups_by_name:
        groups_by_name[group_name].sessions.extend([session])
    else:
        group = api_pb2.SessionGroup(name=group_name, sessions=[session], monitor_url=start_info.monitor_url)
        for key, value in start_info.hparams.items():
            if not json_format_compat.is_serializable_value(value):
                continue
            group.hparams[key].CopyFrom(value)
        groups_by_name[group_name] = group