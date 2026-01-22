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
def _peek_obs(self):
    multi_obs = {}
    if not self.done:
        logger.debug('Peeking the clients.')
        peek_message = '<Peek/>'
        multi_done = True
        for actor_name, instance in zip(self.task.agent_names, self.instances):
            start_time = time.time()
            comms.send_message(instance.client_socket, peek_message.encode())
            obs = comms.recv_message(instance.client_socket)
            info = comms.recv_message(instance.client_socket).decode('utf-8')
            reply = comms.recv_message(instance.client_socket)
            done, = struct.unpack('!b', reply)
            self.has_finished[actor_name] = self.has_finished[actor_name] or done
            multi_done = multi_done and done == 1
            if obs is None or len(obs) == 0:
                if time.time() - start_time > MAX_WAIT:
                    instance.client_socket.close()
                    instance.client_socket = None
                    raise MissionInitException('too long waiting for first observation')
                time.sleep(0.1)
            multi_obs[actor_name], _ = self._process_observation(actor_name, obs, info)
        self.done = multi_done
        if self.done:
            raise RuntimeError('Something went wrong resetting the environment! `done` was true on first frame.')
    return multi_obs