import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def _statstream(self):
    if self.stream:
        sres = os.fstat(self.stream.fileno())
        self.dev, self.ino = (sres[ST_DEV], sres[ST_INO])