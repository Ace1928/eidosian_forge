class ShutDownImminentException(Exception):

    def __init__(self, msg, extra_info):
        self.extra_info = extra_info
        super().__init__(msg)