import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _new_tag(self):
    tag = self.tagpre + bytes(str(self.tagnum), self._encoding)
    self.tagnum = self.tagnum + 1
    self.tagged_commands[tag] = None
    return tag