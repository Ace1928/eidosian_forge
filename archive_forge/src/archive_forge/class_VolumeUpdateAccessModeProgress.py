class VolumeUpdateAccessModeProgress(object):

    def __init__(self, task_complete=False, read_only=None):
        self.called = task_complete
        self.read_only = read_only