def _maybe_resume_protocol(self):
    if self._protocol_paused and self.get_write_buffer_size() <= self._low_water:
        self._protocol_paused = False
        try:
            self._protocol.resume_writing()
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._loop.call_exception_handler({'message': 'protocol.resume_writing() failed', 'exception': exc, 'transport': self, 'protocol': self._protocol})