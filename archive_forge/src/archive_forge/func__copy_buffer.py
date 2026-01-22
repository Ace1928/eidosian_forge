from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def _copy_buffer(self, episode: 'MultiAgentEpisode') -> None:
    """Writes values from one buffer to another."""
    for agent_id, agent_buffer in episode.agent_buffers.items():
        for buffer_name, buffer in agent_buffer.items():
            if buffer.full():
                item = buffer.get_nowait()
                if self.agent_buffers[agent_id][buffer_name].full():
                    self.agent_buffers[agent_id][buffer_name].get_nowait()
                self.agent_buffers[agent_id][buffer_name].put_nowait(item)
                buffer.put_nowait(item)
            elif self.agent_buffers[agent_id][buffer_name].full():
                self.agent_buffers[agent_id][buffer_name].get_nowait()