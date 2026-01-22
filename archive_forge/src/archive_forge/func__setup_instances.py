import traceback
from copy import deepcopy
import json
import logging
from minerl.env.comms import retry
from minerl.env.exceptions import MissionInitException
import os
from minerl.herobraine.wrapper import EnvWrapper
import struct
from minerl.env.malmo import InstanceManager, MinecraftInstance, launch_queue_logger_thread, malmo_version
import uuid
import coloredlogs
import gym
import socket
import time
from lxml import etree
from minerl.env import comms
import xmltodict
from concurrent.futures import ThreadPoolExecutor
import cv2
from minerl.herobraine.env_spec import EnvSpec
from typing import Any, Callable, Dict, List, Optional, Tuple
def _setup_instances(self) -> None:
    """Sets up the instances for the environment 
        """
    num_instances_to_start = self.task.agent_count - len(self.instances)
    num_old_instances = len(self.instances)
    instance_futures = []
    if num_instances_to_start > 0:
        with ThreadPoolExecutor(max_workers=num_instances_to_start) as tpe:
            for _ in range(num_instances_to_start):
                instance_futures.append(tpe.submit(self._get_new_instance))
        self.instances.extend([f.result() for f in instance_futures])
        self.instances = self.instances[:self.task.agent_count]
    if self._refresh_inst_every is not None and self._inst_setup_cntr % self._refresh_inst_every == 0:
        for i in reversed(range(num_old_instances)):
            self.instances[i].kill()
            self.instances[i] = self._get_new_instance(instance_id=self.instances[i].instance_id)
    self._inst_setup_cntr += 1
    for instance in reversed(self.instances):
        self._TO_MOVE_clean_connection(instance)
        self._TO_MOVE_create_connection(instance)
        self._TO_MOVE_quit_current_episode(instance)