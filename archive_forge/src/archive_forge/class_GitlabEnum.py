from enum import Enum, IntEnum
from gitlab._version import __title__, __version__
class GitlabEnum(str, Enum):
    """An enum mixed in with str to make it JSON-serializable."""