from __future__ import annotations
import codecs
import dataclasses
import pathlib
import re
@dataclasses.dataclass(frozen=True)
class MountEntry:
    """A single mount info entry parsed from '/proc/{pid}/mountinfo' in the proc filesystem."""
    mount_id: int
    parent_id: int
    device_major: int
    device_minor: int
    root: pathlib.PurePosixPath
    path: pathlib.PurePosixPath
    options: tuple[str, ...]
    fields: tuple[str, ...]
    type: str
    source: pathlib.PurePosixPath
    super_options: tuple[str, ...]

    @classmethod
    def parse(cls, value: str) -> MountEntry:
        """Parse the given mount info line from the proc filesystem and return a mount entry."""
        mount_id, parent_id, device_major_minor, root, path, options, *remainder = value.split(' ')
        fields = remainder[:-4]
        separator, mtype, source, super_options = remainder[-4:]
        assert separator == '-'
        device_major, device_minor = device_major_minor.split(':')
        return cls(mount_id=int(mount_id), parent_id=int(parent_id), device_major=int(device_major), device_minor=int(device_minor), root=_decode_path(root), path=_decode_path(path), options=tuple(options.split(',')), fields=tuple(fields), type=mtype, source=_decode_path(source), super_options=tuple(super_options.split(',')))

    @classmethod
    def loads(cls, value: str) -> tuple[MountEntry, ...]:
        """Parse the given output from the proc filesystem and return a tuple of mount info entries."""
        return tuple((cls.parse(line) for line in value.splitlines()))