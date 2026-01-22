class TimeLimitExceeded(Exception):
    """The time limit has been exceeded and the job has been terminated."""

    def __str__(self):
        return 'TimeLimitExceeded%s' % (self.args,)