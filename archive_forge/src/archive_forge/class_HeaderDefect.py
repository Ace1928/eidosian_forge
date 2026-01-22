class HeaderDefect(MessageDefect):
    """Base class for a header defect."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)