import ast
import cmd
import signal
import socket
import sys
import termios
from os_ken import cfg
from os_ken.lib import rpc
notify <peer> <method> <params>
        send a msgpack-rpc notification.
        <params> is a python code snippet, it should be eval'ed to a list.
        