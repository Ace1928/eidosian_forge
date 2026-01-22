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
def _TO_MOVE_find_ip_and_port(self, instance: MinecraftInstance, token: str) -> Tuple[str, str]:
    sock = instance.client_socket
    port = 0
    tries = 0
    start_time = time.time()
    logger.info('Attempting to find_ip: {instance}'.format(instance=instance))
    while port == 0 and time.time() - start_time <= MAX_WAIT:
        comms.send_message(sock, ('<Find>' + token + '</Find>').encode())
        reply = comms.recv_message(sock)
        port, = struct.unpack('!I', reply)
        tries += 1
        time.sleep(0.1)
    if port == 0:
        raise Exception('Failed to find master server port!')
    self.integratedServerPort = port
    logger.warning('MineRL agent is public, connect on port {} with Minecraft 1.11'.format(port))
    return (instance.host, str(port))