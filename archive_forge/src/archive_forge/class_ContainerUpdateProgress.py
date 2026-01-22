class ContainerUpdateProgress(UpdateProgressBase):

    def __init__(self, container_id, handler, complete=False, called=False, handler_extra=None, checker_extra=None):
        super(ContainerUpdateProgress, self).__init__(container_id, handler, complete=complete, called=called, handler_extra=handler_extra, checker_extra=checker_extra)
        self.container_id = container_id