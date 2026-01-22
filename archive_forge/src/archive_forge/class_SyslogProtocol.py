from __future__ import (absolute_import, division, print_function)
import asyncio
import json
import logging
from typing import Any, Dict
import re
class SyslogProtocol(asyncio.DatagramProtocol):

    def __init__(self, edaQueue):
        super().__init__()
        self.edaQueue = edaQueue

    def connection_made(self, transport) -> 'Used by asyncio':
        self.transport = transport

    def datagram_received(self, data, addr):
        asyncio.get_event_loop().create_task(self.datagram_received_async(data, addr))

    async def datagram_received_async(self, indata, addr) -> 'Main entrypoint for processing message':
        logger = logging.getLogger()
        rcvdata = indata.decode()
        logger.info('Received Syslog message: %s', rcvdata)
        data = parse(rcvdata)
        if data is None:
            try:
                value = rcvdata[rcvdata.index('{'):len(rcvdata)]
                data = json.loads(value)
            except json.decoder.JSONDecodeError as jerror:
                logger.error(jerror)
                data = rcvdata
            except UnicodeError as e:
                logger.error(e)
        if data:
            queue = self.edaQueue
            await queue.put({'cyberark': data})