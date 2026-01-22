from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
class PSTracker(object):
    """
    Tracker module for PS
    """

    def __init__(self, hostIP, cmd, port=9091, port_end=9999, envs=None):
        """
        Starts the PS scheduler
        """
        self.cmd = cmd
        if cmd is None:
            return
        envs = {} if envs is None else envs
        self.hostIP = hostIP
        sock = socket.socket(get_family(hostIP), socket.SOCK_STREAM)
        for port in range(port, port_end):
            try:
                sock.bind(('', port))
                self.port = port
                sock.close()
                break
            except socket.error:
                continue
        env = os.environ.copy()
        env['DMLC_ROLE'] = 'scheduler'
        env['DMLC_PS_ROOT_URI'] = str(self.hostIP)
        env['DMLC_PS_ROOT_PORT'] = str(self.port)
        for k, v in envs.items():
            env[k] = str(v)
        self.thread = Thread(target=lambda: subprocess.check_call(self.cmd, env=env, shell=True, executable='/bin/bash'), args=())
        self.thread.setDaemon(True)
        self.thread.start()

    def join(self):
        if self.cmd is not None:
            while self.thread.isAlive():
                self.thread.join(100)

    def slave_envs(self):
        if self.cmd is None:
            return {}
        else:
            return {'DMLC_PS_ROOT_URI': self.hostIP, 'DMLC_PS_ROOT_PORT': self.port}

    def alive(self):
        if self.cmd is not None:
            return self.thread.isAlive()
        else:
            return False