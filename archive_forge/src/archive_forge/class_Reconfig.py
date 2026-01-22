from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class Reconfig(Subconstruct):
    """
    Reconfigures a subconstruct. Reconfig can be used to change the name and
    set and clear flags of the inner subcon.

    Parameters:
    * name - the new name
    * subcon - the subcon to reconfigure
    * setflags - the flags to set (default is 0)
    * clearflags - the flags to clear (default is 0)

    Example:
    Reconfig("foo", UBInt8("bar"))
    """
    __slots__ = []

    def __init__(self, name, subcon, setflags=0, clearflags=0):
        Construct.__init__(self, name, subcon.conflags)
        self.subcon = subcon
        self._set_flag(setflags)
        self._clear_flag(clearflags)