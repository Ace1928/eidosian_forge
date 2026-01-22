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
def _TO_MOVE_quit_current_episode(self, instance: MinecraftInstance) -> None:
    has_quit = False
    logger.info('Attempting to quit: {instance}'.format(instance=instance))
    comms.send_message(instance.client_socket, '<Quit/>'.encode())
    reply = comms.recv_message(instance.client_socket)
    ok, = struct.unpack('!I', reply)
    has_quit = not ok == 0