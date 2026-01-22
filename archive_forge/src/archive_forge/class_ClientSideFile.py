import asyncio
import copy
import io
import os
import sys
import IPython
import traitlets
import ipyvuetify as v
class ClientSideFile(io.RawIOBase):

    def __init__(self, widget, file_index, timeout=30):
        global chunk_listener_id
        self.id = chunk_listener_id
        self.widget = widget
        self.version = widget.version
        self.file_index = file_index
        self.timeout = timeout
        self.valid = True
        self.offset = 0
        self.size = widget.file_info[file_index]['size']
        self.chunk_queue = []
        widget.chunk_listeners[self.id] = self
        chunk_listener_id += 1
        self.waits = 0

    def handle_chunk(self, content, buffer):
        content['buffer'] = buffer
        self.chunk_queue.append(content)

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.offset = offset
        elif whence == io.SEEK_CUR:
            self.offset = self.offset + offset
        elif whence == io.SEEK_END:
            self.offset = self.size + offset
        else:
            raise ValueError(f'whence {whence} invalid')

    def tell(self):
        return self.offset

    def readinto(self, buffer):
        if not self.valid:
            raise Exception('Invalid file state')
        mem = memoryview(buffer)
        remaining = max(0, self.size - self.offset)
        size = min(len(buffer), remaining)
        self.widget.send({'method': 'read', 'args': [{'file_index': self.file_index, 'offset': self.offset, 'length': size, 'id': self.id}]})
        sleep_interval = 0.01
        max_iterations = self.timeout / sleep_interval

        async def read_all():
            bytes_read = 0
            while bytes_read < size:
                iterations = 0
                while not self.chunk_queue:
                    iterations += 1
                    if self.version != self.widget.version:
                        self.valid = False
                        raise Exception('File changed')
                    if iterations > max_iterations:
                        self.valid = False
                        raise Exception('Timeout')
                    await asyncio.sleep(sleep_interval)
                    await process_messages()
                self.waits += iterations
                chunk = self.chunk_queue[0]
                chunk_size = chunk['length']
                mem[bytes_read:bytes_read + chunk_size] = chunk['buffer']
                self.chunk_queue.pop(0)
                bytes_read += chunk_size
                self.offset += chunk_size
                self.widget.update_stats(self.file_index, chunk_size)
                await process_messages()

        def has_event_loop():
            try:
                asyncio.get_event_loop()
                return True
            except RuntimeError:
                return False
        if has_event_loop():
            if not has_nest_asyncio:
                raise RuntimeError("nest_asyncio is required for FileInput when an event loop is already running in the current thread, please run 'pip install nest_asyncio'.")
            else:
                nest_asyncio.apply()
        asyncio.run(read_all())
        return size

    def readall(self):
        return self.read(self.size - self.offset)