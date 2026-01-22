import logging
import queue
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from typing import (
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv, convert_to_base_env
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.env_runner_v2 import (
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.offline import InputReader
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy, make_action_immutable
from ray.rllib.utils.spaces.space_utils import clip_action, unbatch, unsquash_action
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _process_policy_eval_results(*, to_eval: Dict[PolicyID, List[_PolicyEvalData]], eval_results: Dict[PolicyID, Tuple[TensorStructType, StateBatch, dict]], active_episodes: Dict[EnvID, Episode], active_envs: Set[int], off_policy_actions: MultiEnvDict, policies: Dict[PolicyID, Policy], normalize_actions: bool, clip_actions: bool) -> Dict[EnvID, Dict[AgentID, EnvActionType]]:
    """Process the output of policy neural network evaluation.

    Records policy evaluation results into the given episode objects and
    returns replies to send back to agents in the env.

    Args:
        to_eval: Mapping of policy IDs to lists of _PolicyEvalData objects.
        eval_results: Mapping of policy IDs to list of
            actions, rnn-out states, extra-action-fetches dicts.
        active_episodes: Mapping from episode ID to currently ongoing
            Episode object.
        active_envs: Set of non-terminated env ids.
        off_policy_actions: Doubly keyed dict of env-ids -> agent ids ->
            off-policy-action, returned by a `BaseEnv.poll()` call.
        policies: Mapping from policy ID to Policy.
        normalize_actions: Whether to normalize actions to the action
            space's bounds.
        clip_actions: Whether to clip actions to the action space's bounds.

    Returns:
        Nested dict of env id -> agent id -> actions to be sent to
        Env (np.ndarrays).
    """
    actions_to_send: Dict[EnvID, Dict[AgentID, EnvActionType]] = defaultdict(dict)
    for env_id in active_envs:
        actions_to_send[env_id] = {}
    for policy_id, eval_data in to_eval.items():
        actions: TensorStructType = eval_results[policy_id][0]
        actions = convert_to_numpy(actions)
        rnn_out_cols: StateBatch = eval_results[policy_id][1]
        extra_action_out_cols: dict = eval_results[policy_id][2]
        if isinstance(actions, list):
            actions = np.array(actions)
        for f_i, column in enumerate(rnn_out_cols):
            extra_action_out_cols['state_out_{}'.format(f_i)] = column
        policy: Policy = _get_or_raise(policies, policy_id)
        actions: List[EnvActionType] = unbatch(actions)
        for i, action in enumerate(actions):
            if normalize_actions:
                action_to_send = unsquash_action(action, policy.action_space_struct)
            elif clip_actions:
                action_to_send = clip_action(action, policy.action_space_struct)
            else:
                action_to_send = action
            env_id: int = eval_data[i].env_id
            agent_id: AgentID = eval_data[i].agent_id
            episode: Episode = active_episodes[env_id]
            _assert_episode_not_faulty(episode)
            episode._set_rnn_state(agent_id, tree.map_structure(lambda x: x[i], rnn_out_cols))
            episode._set_last_extra_action_outs(agent_id, tree.map_structure(lambda x: x[i], extra_action_out_cols))
            if env_id in off_policy_actions and agent_id in off_policy_actions[env_id]:
                episode._set_last_action(agent_id, off_policy_actions[env_id][agent_id])
            else:
                episode._set_last_action(agent_id, action)
            assert agent_id not in actions_to_send[env_id]
            tree.traverse(make_action_immutable, action_to_send, top_down=False)
            actions_to_send[env_id][agent_id] = action_to_send
    return actions_to_send