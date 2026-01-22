from collections import namedtuple
class Exchange:
    """AMQ Exchange class."""
    CLASS_ID = 40
    Declare = (40, 10)
    DeclareOk = (40, 11)
    Delete = (40, 20)
    DeleteOk = (40, 21)
    Bind = (40, 30)
    BindOk = (40, 31)
    Unbind = (40, 40)
    UnbindOk = (40, 51)