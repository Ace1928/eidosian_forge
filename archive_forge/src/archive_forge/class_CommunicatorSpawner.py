import asyncio
import subprocess
import sys
import pytest
from tornado.queues import Queue
from jupyter_lsp.stdio import LspStdIoReader
from time import sleep
class CommunicatorSpawner:

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path

    def spawn_writer(self, message: str, repeats: int=1, interval=None, add_excess=False):
        length = len(message) * repeats
        commands_file = self.tmp_path / 'writer.py'
        commands_file.write_text(WRITER_TEMPLATE.format(length=length, repeats=repeats, interval=interval or 0, message=message, add_excess=add_excess))
        return subprocess.Popen([sys.executable, '-u', str(commands_file)], stdout=subprocess.PIPE, bufsize=0)