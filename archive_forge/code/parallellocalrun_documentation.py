from subprocess import Popen, PIPE
import os
import time
from ase.io import write, read
 Checks if any relaxations are done and load in the structure
            from the traj file. 