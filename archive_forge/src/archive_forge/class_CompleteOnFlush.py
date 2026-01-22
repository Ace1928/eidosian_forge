import duet
import duet.impl as impl
class CompleteOnFlush(duet.BufferedFuture):

    def __init__(self):
        super().__init__()
        self.flushed = False

    def flush(self):
        self.flushed = True
        self.set_result(None)