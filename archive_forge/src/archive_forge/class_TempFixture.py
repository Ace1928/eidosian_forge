import io
import os
import pty
import tempfile
import fixtures  # type: ignore
import typing
class TempFixture(fixtures.Fixture):

    def _setUp(self) -> None:
        self.stream = typing.cast(typing.TextIO, tempfile.TemporaryFile('w'))

        def close() -> None:
            try:
                self.stream.close()
            except ValueError:
                pass
        self.addCleanup(close)