from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def _generate_ts_mapping(self, observations: List[MultiAgentDict]) -> MultiAgentDict:
    """Generates a timestep mapping to local agent timesteps.

        This helps us to keep track of which agent stepped at
        which global (environment) timestep.
        Note that the local (agent) timestep is given by the index
        of the list for each agent.

        Args:
            observations: A list of observations.Each observations maps agent
                ids to their corresponding observation.

        Returns: A dictionary mapping agents to time index lists. The latter
            contain the global (environment) timesteps at which the agent
            stepped (was ready).
        """
    if len(self._agent_ids) > 0:
        global_t_to_local_t = {agent: _IndexMapping() for agent in self._agent_ids}
        if observations:
            for t, agent_map in enumerate(observations):
                for agent_id in agent_map:
                    global_t_to_local_t[agent_id].append(t + self.ts_carriage_return)
        else:
            global_t_to_local_t = {}
    else:
        global_t_to_local_t = {}
    return global_t_to_local_t