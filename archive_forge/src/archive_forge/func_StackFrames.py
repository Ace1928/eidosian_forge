import threading
from tensorboard import errors
def StackFrames(self, run, stack_frame_ids):
    runs = self.Runs()
    if run not in runs:
        return None
    stack_frames = []
    for stack_frame_id in stack_frame_ids:
        if stack_frame_id not in self._reader._stack_frame_by_id:
            raise errors.NotFoundError('Cannot find stack frame with ID %s' % stack_frame_id)
        stack_frames.append(self._reader._stack_frame_by_id[stack_frame_id])
    return {'stack_frames': stack_frames}