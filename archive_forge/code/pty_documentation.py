from select import select
import os
import sys
import tty
from os import close, waitpid
from tty import setraw, tcgetattr, tcsetattr
Create a spawned process.