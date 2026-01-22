import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _simple_command(self, name, *args):
    return self._command_complete(name, self._command(name, *args))