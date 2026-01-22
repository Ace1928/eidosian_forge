from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class RefLogEntry(Tuple[str, str, Actor, Tuple[int, int], str]):
    """Named tuple allowing easy access to the revlog data fields."""
    _re_hexsha_only = re.compile('^[0-9A-Fa-f]{40}$')
    __slots__ = ()

    def __repr__(self) -> str:
        """Representation of ourselves in git reflog format."""
        return self.format()

    def format(self) -> str:
        """:return: A string suitable to be placed in a reflog file."""
        act = self.actor
        time = self.time
        return '{} {} {} <{}> {!s} {}\t{}\n'.format(self.oldhexsha, self.newhexsha, act.name, act.email, time[0], altz_to_utctz_str(time[1]), self.message)

    @property
    def oldhexsha(self) -> str:
        """The hexsha to the commit the ref pointed to before the change."""
        return self[0]

    @property
    def newhexsha(self) -> str:
        """The hexsha to the commit the ref now points to, after the change."""
        return self[1]

    @property
    def actor(self) -> Actor:
        """Actor instance, providing access."""
        return self[2]

    @property
    def time(self) -> Tuple[int, int]:
        """time as tuple:

        * [0] = ``int(time)``
        * [1] = ``int(timezone_offset)`` in :attr:`time.altzone` format
        """
        return self[3]

    @property
    def message(self) -> str:
        """Message describing the operation that acted on the reference."""
        return self[4]

    @classmethod
    def new(cls, oldhexsha: str, newhexsha: str, actor: Actor, time: int, tz_offset: int, message: str) -> 'RefLogEntry':
        """:return: New instance of a RefLogEntry"""
        if not isinstance(actor, Actor):
            raise ValueError('Need actor instance, got %s' % actor)
        return RefLogEntry((oldhexsha, newhexsha, actor, (time, tz_offset), message))

    @classmethod
    def from_line(cls, line: bytes) -> 'RefLogEntry':
        """:return: New RefLogEntry instance from the given revlog line.

        :param line: Line bytes without trailing newline

        :raise ValueError: If `line` could not be parsed
        """
        line_str = line.decode(defenc)
        fields = line_str.split('\t', 1)
        if len(fields) == 1:
            info, msg = (fields[0], None)
        elif len(fields) == 2:
            info, msg = fields
        else:
            raise ValueError('Line must have up to two TAB-separated fields. Got %s' % repr(line_str))
        oldhexsha = info[:40]
        newhexsha = info[41:81]
        for hexsha in (oldhexsha, newhexsha):
            if not cls._re_hexsha_only.match(hexsha):
                raise ValueError('Invalid hexsha: %r' % (hexsha,))
        email_end = info.find('>', 82)
        if email_end == -1:
            raise ValueError('Missing token: >')
        actor = Actor._from_string(info[82:email_end + 1])
        time, tz_offset = parse_date(info[email_end + 2:])
        return RefLogEntry((oldhexsha, newhexsha, actor, (time, tz_offset), msg))