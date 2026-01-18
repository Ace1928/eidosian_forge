import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
Ensure FuncResult sets its result and exception attributes