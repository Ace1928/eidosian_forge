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
def _TO_MOVE_clean_connection(self, instance: MinecraftInstance) -> None:
    """
        Cleans the conenction with a given instance.
        """
    try:
        if instance.client_socket:
            try:
                comms.send_message(instance.client_socket, '<Disconnect/>'.encode())
            except:
                pass
            instance.client_socket.shutdown(socket.SHUT_RDWR)
            instance.client_socket.close()
    except (BrokenPipeError, OSError, socket.error):
        pass
        instance.client_socket = None