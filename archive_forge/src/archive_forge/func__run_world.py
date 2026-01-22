import logging
import time
import datetime
from concurrent import futures
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
def _run_world(self, task, world_name, agents):
    """
        Run a world until completion.

        :param task:
            TaskState. State of the given task.
        :param world_name:
            string. The name of the world in the module file
        :param agents:
            list. A list of agents that should be in the world.

        :return:
            ret_val: last output of world's parley function. Return None if ERROR
            world_data: data attribute of world, if it has one
        """
    ret_val = None
    world_generator = utils.get_world_fn_attr(self._world_module, world_name, 'generate_world')
    world = world_generator(self.opt, agents)
    task.world = world
    while not world.episode_done() and (not self.system_done):
        ret_val = world.parley()
        time.sleep(0.3)
    world.shutdown()
    world_data = world.data if hasattr(world, 'data') else {}
    return (ret_val, world_data)