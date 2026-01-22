import getopt
import os
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
from os_ken.ofproto import ofproto_protocol
class MyVerboseHandler(MyHandler):
    verbose = True