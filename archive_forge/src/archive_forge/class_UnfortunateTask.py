import os
from taskflow.patterns import linear_flow as lf
from taskflow import task
class UnfortunateTask(task.Task):

    def execute(self):
        print('executing %s' % self)
        boom = os.environ.get('BOOM')
        if boom:
            print('> Critical error: boom = %s' % boom)
            raise SystemExit()
        else:
            print('> this time not exiting')