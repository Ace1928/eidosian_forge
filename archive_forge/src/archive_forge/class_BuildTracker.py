import contextlib
import hashlib
import logging
import os
from types import TracebackType
from typing import Dict, Generator, Optional, Set, Type, Union
from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.temp_dir import TempDirectory
class BuildTracker:
    """Ensure that an sdist cannot request itself as a setup requirement.

    When an sdist is prepared, it identifies its setup requirements in the
    context of ``BuildTracker.track()``. If a requirement shows up recursively, this
    raises an exception.

    This stops fork bombs embedded in malicious packages."""

    def __init__(self, root: str) -> None:
        self._root = root
        self._entries: Dict[TrackerId, InstallRequirement] = {}
        logger.debug('Created build tracker: %s', self._root)

    def __enter__(self) -> 'BuildTracker':
        logger.debug('Entered build tracker: %s', self._root)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.cleanup()

    def _entry_path(self, key: TrackerId) -> str:
        hashed = hashlib.sha224(key.encode()).hexdigest()
        return os.path.join(self._root, hashed)

    def add(self, req: InstallRequirement, key: TrackerId) -> None:
        """Add an InstallRequirement to build tracking."""
        entry_path = self._entry_path(key)
        try:
            with open(entry_path) as fp:
                contents = fp.read()
        except FileNotFoundError:
            pass
        else:
            message = '{} is already being built: {}'.format(req.link, contents)
            raise LookupError(message)
        assert key not in self._entries
        with open(entry_path, 'w', encoding='utf-8') as fp:
            fp.write(str(req))
        self._entries[key] = req
        logger.debug('Added %s to build tracker %r', req, self._root)

    def remove(self, req: InstallRequirement, key: TrackerId) -> None:
        """Remove an InstallRequirement from build tracking."""
        os.unlink(self._entry_path(key))
        del self._entries[key]
        logger.debug('Removed %s from build tracker %r', req, self._root)

    def cleanup(self) -> None:
        for key, req in list(self._entries.items()):
            self.remove(req, key)
        logger.debug('Removed build tracker: %r', self._root)

    @contextlib.contextmanager
    def track(self, req: InstallRequirement, key: str) -> Generator[None, None, None]:
        """Ensure that `key` cannot install itself as a setup requirement.

        :raises LookupError: If `key` was already provided in a parent invocation of
                             the context introduced by this method."""
        tracker_id = TrackerId(key)
        self.add(req, tracker_id)
        yield
        self.remove(req, tracker_id)