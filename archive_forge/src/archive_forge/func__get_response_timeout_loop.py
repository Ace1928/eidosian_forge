import time
from typing import List, Optional
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.worlds import World
@staticmethod
def _get_response_timeout_loop(agent: Agent, world: World, timeout: int=DEFAULT_TIMEOUT, timeout_msg: str='You have timed out') -> Optional[Message]:
    """
        Get a response from the agent.

        :param agent:
            agent who is acting
        :param world:
            world in which agent is acting
        :param timeout:
            timeout in secs
        :param timeout_msg:
            what to say to agent when they timeout

        :return response:
            Response if given, else None
        """
    a = TimeoutUtils.get_timeout_act(agent, timeout)
    if a is None:
        world.episodeDone = True
        agent.observe({'id': '', 'text': timeout_msg})
        return None
    if (a.get('text', '') or '').upper() == 'EXIT':
        world.episodeDone = True
        return None
    return a