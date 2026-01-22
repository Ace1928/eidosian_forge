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
def _session_groups_from_data_provider(self):
    """Constructs lists of SessionGroups based on DataProvider results."""
    filters = _build_data_provider_filters(self._request.col_params)
    sort = _build_data_provider_sort(self._request.col_params)
    response = self._backend_context.session_groups_from_data_provider(self._request_context, self._experiment_id, filters, sort)
    session_groups = []
    for provider_group in response:
        sessions = [api_pb2.Session(name=f'{s.experiment_id}/{s.run}') for s in provider_group.sessions]
        name = f'{provider_group.root.experiment_id}/{provider_group.root.run}' if provider_group.root.run else provider_group.root.experiment_id
        session_group = api_pb2.SessionGroup(name=name, sessions=sessions)
        for provider_hparam in provider_group.hyperparameter_values:
            hparam = session_group.hparams[provider_hparam.hyperparameter_name]
            if provider_hparam.domain_type == provider.HyperparameterDomainType.DISCRETE_STRING:
                hparam.string_value = provider_hparam.value
            elif provider_hparam.domain_type in [provider.HyperparameterDomainType.DISCRETE_FLOAT, provider.HyperparameterDomainType.INTERVAL]:
                hparam.number_value = provider_hparam.value
            elif provider_hparam.domain_type == provider.HyperparameterDomainType.DISCRETE_BOOL:
                hparam.bool_value = provider_hparam.value
        session_groups.append(session_group)
    return session_groups