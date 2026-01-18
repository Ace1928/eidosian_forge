import random
import time
import math
import os
from collections import deque
from kivy.tests import UnitTestTouch
def touch_up(self, *args):
    self.eventloop._dispatch_input('end', self)