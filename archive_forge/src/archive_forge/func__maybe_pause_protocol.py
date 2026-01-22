def _maybe_pause_protocol(self):
    size = self.get_write_buffer_size()
    if size <= self._high_water:
        return
    if not self._protocol_paused:
        self._protocol_paused = True
        try:
            self._protocol.pause_writing()
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._loop.call_exception_handler({'message': 'protocol.pause_writing() failed', 'exception': exc, 'transport': self, 'protocol': self._protocol})