class Critical(CheckMessage):

    def __init__(self, *args, **kwargs):
        super().__init__(CRITICAL, *args, **kwargs)