import argparse
import logging
import socket
import struct
import sys
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple, Union
def worker_envs(self) -> Dict[str, Union[str, int]]:
    """
        get environment variables for workers
        can be passed in as args or envs
        """
    return {'DMLC_TRACKER_URI': self.host_ip, 'DMLC_TRACKER_PORT': self.port}