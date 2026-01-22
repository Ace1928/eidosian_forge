from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def FMT(self, fmt, *args, **kwargs):
    return self._enc(fmt.format(*args, **dict(kwargs, IN=self.IN, INp=self.INp)))